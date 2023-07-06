#![allow(non_snake_case)]
// #![feature(associated_type_defaults)]
// #![feature(more_qualified_paths)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(non_upper_case_globals)]

pub mod php;

pub mod r1cslite;

pub mod matrixutils;

pub mod comms;

pub mod evalpolyproofs;

#[macro_use]
pub mod phplite2;

#[cfg(test)]
pub mod test;

use ark_ff::ToBytes;
use itertools::Itertools;
use itertools::Zip;
use std::cmp;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Error, Read, Write};
use std::iter::zip;

// Import Randomness
use rand::thread_rng;
use ark_std::rand::Rng;
use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};
use rand::prelude::IteratorRandom;

use ark_std::marker::PhantomData;
use ark_std::{end_timer, start_timer};
use std::time::Instant;


use ark_ec::{PairingEngine, ProjectiveCurve};

use ark_ff::{to_bytes, FftField, Field, One, PrimeField, Zero};

extern crate rand_chacha;

use ark_marlin::rng::FiatShamirRng;
use ark_marlin::rng::SimpleHashFiatShamirRng;
use ark_marlin::*;
use blake2::Blake2s;
use rand_chacha::ChaChaRng;

use ark_poly::{
    domain::EvaluationDomain,
    evaluations::univariate::Evaluations,
    multivariate::{SparsePolynomial as SPolynomial, SparseTerm, Term},
    polynomial::Polynomial,
    univariate::{DenseOrSparsePolynomial, DensePolynomial as DPolynomial},
    GeneralEvaluationDomain, MVPolynomial, UVPolynomial,
};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};

use crate::comms::*;
use crate::evalpolyproofs::*;
use crate::matrixutils::*;
use crate::php::*;
use crate::phplite2::*;

use crate::r1cslite::{example_simple, simple_matrixL, simple_matrixR};
use crate::r1cslite::{example_simple_v2, simple_matrixL_v2, simple_matrixR_v2, ExampleR1CSLite};

impl From<Error> for CPError {
    fn from(e: Error) -> CPError {
        CPError::BadPrf
    }
}

/// The compiled argument system.
pub struct LunarLite<F: FftField, P: PairingEngine + PairingEngine<Fr = F>, C: CommitmentScheme<P>>(
    #[doc(hidden)] PhantomData<P>,
    #[doc(hidden)] PhantomData<F>,
    #[doc(hidden)] PhantomData<C>,
);

impl<
        F: FftField,
        P: PairingEngine + PairingEngine<Fr = F>,
        C: CommitmentScheme<P> + CPdeg<P, C>,
    > LunarLite<F, P, C>
{
    pub const PROTOCOL_NAME: &'static [u8] = b"LUNARLITE-2022";

    pub fn setup_relation_encoder<R>(
        n_max: usize,
        n_example: usize,
        m: usize,
        l: usize,
        cs_max_degree: usize,
        rng: &mut R,
        id: &str,
        exampler1cslite: ExampleR1CSLite<F>,
    ) -> (
        (ProverPP<F>, VerifierPP<F>, C::CS_PP),
        (View<F>, View<F>, CView<P, C>),
        Vec<P::G2Projective>,
    )
    where
        R: rand_core::RngCore,
    {
        // let now = Instant::now();

        // Setup of the parameters
        // For now, the PHP only take as input squared matrices of dimension a power of two
        // A function will come later to convert any PHP input to the appropriate size
        // We want n to be a power of two greater than n_example
        let n_next_pow2 = (3 * n_example).next_power_of_two();

        assert!(
            n_next_pow2 <= n_max,
            "n value for example generation larger than n_max"
        );

        // Create new examples with random parameters
        // Inialization of the example

        let ExampleR1CSLite {
            matrices: (L, R),
            polynomials: (a_prime_poly, b_prime_poly),
            vector: x_vec,
        } = exampler1cslite;

        let cs = CSLiteParams::new(n_next_pow2, m, l, L, R);

        // let elapsed_time = now.elapsed();
        // println!("Setup example generation took {} microsec.", elapsed_time.as_micros());
        let now = Instant::now();

        // initialize parameters for prover
        let prover_PP = ProverPP {
            cs: cs.clone(),
            x_vec: x_vec.clone(),
            a_prime_poly,
            b_prime_poly,
        };

        // initialize parameters for verifier
        let verifier_PP = VerifierPP {
            cs: cs.clone(),
            x_vec: x_vec.clone(),
        };

        // Generate relation encoder values
        // Need to setup a way to ingrate them the the prover and verifier param
        let (n, m, l) = prover_PP.cs.get_params();
        let domain_k = GeneralEvaluationDomain::<F>::new(m).unwrap();
        let domain_k_size = domain_k.size();

        let mut oracles_vec = R1CSLite2::setup(&prover_PP.cs);
        let v_crL = oracles_vec.remove(0).get_matrix().unwrap();
        let v_crR = oracles_vec.remove(0).get_matrix().unwrap();
        let cr = oracles_vec.remove(0).get_matrix().unwrap();
        let cr_p = oracles_vec.remove(0).get_matrix().unwrap();

        let elapsed_time = now.elapsed();
        // println!(
        //     "Setup R1CSLite2 polys took {} microsec.",
        //     elapsed_time.as_micros()
        // );
        let now = Instant::now();

        /// Parameters specific to R1CSLite2
        // imported from the prover.rs file
        // Are those parameters fixed?
        const b_a: usize = 1;
        const b_b: usize = 2;
        const b_q: usize = 1;
        // const b_r: usize = 1;
        const b_s: usize = 1;

        // Setup of the CS parameters // change with max values for n and m
        let D = cmp::max(
            2 * cs_max_degree + b_a + b_b + 2 * b_q - 3,
            cmp::max(cs_max_degree + b_s + b_q - 1, m),
        );

        let cs_pp: C::CS_PP = C::setup(rng, D);

        let elapsed_time = now.elapsed();
        // println!(
        //     "Setup CS generation took {} ms.",
        //     elapsed_time.as_millis()
        // );
        let now = Instant::now();

        // Degree vector for the degree check proof
        let deg_vec = vec![n - 2, domain_k_size - 2];

        let vk_d = <C as CPdeg<P, C>>::derive(&cs_pp, &deg_vec).unwrap();

        // Creating the view for the prover and the verifier
        // prover view has scalars and polynomials
        // verifer view has scalars
        let mut php_prover_view: View<F> = View::<F>::new();
        let php_verifier_view: View<F> = View::<F>::new();

        // Creation of the prover commitment view
        // Contains the useful commitments for the proofs and verifier checks
        let mut prover_Cview: CView<P, C> = CView::<P, C>::new();

        // Adds relation encoder polynomials to the prover view
        php_prover_view.add_idx_oracle(
            &stringify!(v_crL).to_string(),
            Oracle::OMtxPoly(v_crL.clone()),
        );
        php_prover_view.add_idx_oracle(
            &stringify!(v_crR).to_string(),
            Oracle::OMtxPoly(v_crR.clone()),
        );
        php_prover_view.add_idx_oracle(&stringify!(cr).to_string(), Oracle::OMtxPoly(cr.clone()));
        php_prover_view.add_idx_oracle(
            &stringify!(cr_p).to_string(),
            Oracle::OMtxPoly(cr_p.clone()),
        );

        // Commit to relation encoder polynomials and add them to the commits view
        let commit_RE_polys = commit_oracles_dict_rel::<P, C>(&cs_pp, &php_prover_view.idx_oracles);
        prover_Cview.append_idx_comm_oracles(&commit_RE_polys);

        let commit_RE_cr = prover_Cview.get_idx_oracle_matrix("cr").unwrap();
        let commit_RE_cr_p = prover_Cview.get_idx_oracle_matrix("cr_p").unwrap();
        let commit_RE_vcrL = prover_Cview.get_idx_oracle_matrix("v_crL").unwrap();
        let commit_RE_vcrR = prover_Cview.get_idx_oracle_matrix("v_crR").unwrap();

        let elapsed_time = now.elapsed();
        // println!(
        //     "Setup RE commits took {} ms.",
        //     elapsed_time.as_millis()
        // );

        (
            (prover_PP, verifier_PP, cs_pp),
            (php_prover_view, php_verifier_view, prover_Cview),
            vk_d,
        )
    }

    pub fn prover(
        prover_PP: &ProverPP<F>,
        verifier_PP: &VerifierPP<F>,
        cs_pp: &C::CS_PP,
        php_p_view: &mut View<F>,
        php_v_view: &mut View<F>,
        c_p_view: &mut CView<P, C>,
    ) -> Result<P::G1Projective, CPError>
    where
        C: CommitmentScheme<P, CS_PP = CS2_PP<P>> + CPEval<P, C> + CPdeg<P, C> + CPqeq<P, C>,
    {
        let prover_time = start_timer!(|| "Lunar_pover");

        let commit_RE_cr = c_p_view.get_idx_oracle_matrix("cr").unwrap();
        let commit_RE_cr_p = c_p_view.get_idx_oracle_matrix("cr_p").unwrap();
        let commit_RE_vcrL = c_p_view.get_idx_oracle_matrix("v_crL").unwrap();
        let commit_RE_vcrR = c_p_view.get_idx_oracle_matrix("v_crR").unwrap();

        let flattened_commit_RE_cr: Vec<C::CommT> = commit_RE_cr.into_iter().flatten().collect();
        let flattened_commit_RE_cr_p: Vec<C::CommT> =
            commit_RE_cr_p.into_iter().flatten().collect();
        let flattened_commit_RE_vcrL: Vec<C::CommT> =
            commit_RE_vcrL.into_iter().flatten().collect();
        let flattened_commit_RE_vcrR: Vec<C::CommT> =
            commit_RE_vcrR.into_iter().flatten().collect();

        let mut vec_REL_commit = Vec::new();
        vec_REL_commit.append(&mut flattened_commit_RE_cr.clone());
        vec_REL_commit.append(&mut flattened_commit_RE_cr_p.clone());
        vec_REL_commit.append(&mut flattened_commit_RE_vcrL.clone());
        vec_REL_commit.append(&mut flattened_commit_RE_vcrR.clone());

        let FS_commit_RE =
            <C as CommitmentScheme<P>>::extract_vec_commit_rel(vec_REL_commit.clone())?;

        // Setup the FS RNG using the RE commitments
        let mut fs_rng = SimpleHashFiatShamirRng::<Blake2s, ChaChaRng>::initialize(
            &to_bytes![FS_commit_RE].unwrap(),
        );

        let x_vec = prover_PP.x_vec.clone();
        fs_rng.absorb(&to_bytes![x_vec].unwrap());

        let now = Instant::now();

        R1CSLite2::prover_rounds(
            prover_PP,
            verifier_PP,
            cs_pp,
            php_p_view,
            php_v_view,
            c_p_view,
            &mut fs_rng,
        )?;

        let elapsed_time = now.elapsed();
        // println!("\nProver rounds took {} ms.", elapsed_time.as_millis());

        let now = Instant::now();

        R1CSLite2::prover_precomput(
            prover_PP,
            cs_pp,
            php_p_view,
            php_v_view,
            c_p_view,
            &mut fs_rng,
        );

        let elapsed_time = now.elapsed();
        // println!(
        //     "\nProver precomputations took {} ms.",
        //     elapsed_time.as_millis()
        // );

        let now = Instant::now();

        let proof = R1CSLite2::prover_proofs(prover_PP, cs_pp, php_p_view, c_p_view)?;

        let elapsed_time = now.elapsed();
        // println!("\nProver proof took {} ms.", elapsed_time.as_millis());

        end_timer!(prover_time);

        Ok(proof)
    }

    pub fn verifier(
        verifier_PP: &VerifierPP<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_view: &mut View<P::Fr>,
        prover_commits: &CView<P, C>,
        proof: P::G1Projective,
        vk_d: Vec<P::G2Projective>,
    ) -> Result<(), CPError>
    where
        C: CommitmentScheme<P, CS_PP = CS2_PP<P>> + CPEval<P, C> + CPdeg<P, C> + CPqeq<P, C>,
    {
        let now = Instant::now();

        let commit_RE_cr = prover_commits.get_idx_oracle_matrix("cr").unwrap();
        let commit_RE_cr_p = prover_commits.get_idx_oracle_matrix("cr_p").unwrap();
        let commit_RE_vcrL = prover_commits.get_idx_oracle_matrix("v_crL").unwrap();
        let commit_RE_vcrR = prover_commits.get_idx_oracle_matrix("v_crR").unwrap();

        let flattened_commit_RE_cr: Vec<C::CommT> = commit_RE_cr.into_iter().flatten().collect();
        let flattened_commit_RE_cr_p: Vec<C::CommT> =
            commit_RE_cr_p.into_iter().flatten().collect();
        let flattened_commit_RE_vcrL: Vec<C::CommT> =
            commit_RE_vcrL.into_iter().flatten().collect();
        let flattened_commit_RE_vcrR: Vec<C::CommT> =
            commit_RE_vcrR.into_iter().flatten().collect();

        let mut vec_REL_commit = Vec::new();
        vec_REL_commit.append(&mut flattened_commit_RE_cr.clone());
        vec_REL_commit.append(&mut flattened_commit_RE_cr_p.clone());
        vec_REL_commit.append(&mut flattened_commit_RE_vcrL.clone());
        vec_REL_commit.append(&mut flattened_commit_RE_vcrR.clone());

        let FS_commit_RE =
            <C as CommitmentScheme<P>>::extract_vec_commit_rel(vec_REL_commit.clone())?;

        // Setup the FS RNG using the RE commitments
        let mut fs_rng = SimpleHashFiatShamirRng::<Blake2s, ChaChaRng>::initialize(
            &to_bytes![FS_commit_RE].unwrap(),
        );

        let x_vec = verifier_PP.x_vec.clone();
        fs_rng.absorb(&to_bytes![x_vec].unwrap());

        let elapsed_time = now.elapsed();
        // println!("Verifier extracting RE commit took {} microsec.", elapsed_time.as_micros());
        let now = Instant::now();

        // Create the verifier view
        let mut php_v_view = R1CSLite2::verifier_rounds(
            &verifier_PP,
            prover_view,
            cs_pp,
            &prover_commits,
            &mut fs_rng,
        )?;

        let elapsed_time = now.elapsed();
        // println!("Verifier ROUNDS took {} microsec.", elapsed_time.as_micros());
        let now = Instant::now();

        let ((c_p_prime_eq1, p_eq1_prime_at_y), (cr_coeff, cr_prime_coeff, vcrL_vcrR_coeff)) =
            R1CSLite2::verifier_precompute_for_batched(
                &verifier_PP,
                &mut php_v_view,
                prover_view,
                cs_pp,
                &prover_commits,
                &mut fs_rng,
            );

        let elapsed_time = now.elapsed();
        // println!("Verifier precomputation took {} microsec.", elapsed_time.as_micros());
        let now = Instant::now();

        let _ = R1CSLite2::verifier_checks_batched(
            &verifier_PP,
            cs_pp,
            &mut php_v_view,
            &prover_commits,
            flattened_commit_RE_cr,
            flattened_commit_RE_cr_p,
            flattened_commit_RE_vcrL,
            flattened_commit_RE_vcrR,
            cr_coeff,
            cr_prime_coeff,
            vcrL_vcrR_coeff,
            vk_d,
            proof,
            c_p_prime_eq1,
            p_eq1_prime_at_y,
        )?;

        let elapsed_time = now.elapsed();
        // println!("Verifier CHECKS took {} microsec.", elapsed_time.as_micros());
        let now = Instant::now();

        Ok(())
    }
}

/// The compiled argument system.
pub struct LunarLite2x<F: FftField, P: PairingEngine + PairingEngine<Fr = F>, C: CommitmentScheme<P>>(
    #[doc(hidden)] PhantomData<P>,
    #[doc(hidden)] PhantomData<F>,
    #[doc(hidden)] PhantomData<C>,
);

impl<F: FftField, P: PairingEngine + PairingEngine<Fr = F>, C: CommitmentScheme<P>> LunarLite2x<F, P, C> {
    pub const PROTOCOL_NAME: &'static [u8] = b"LUNAR-2022";

    pub fn setup_relation_encoder<R>(
        n_max: usize,
        n_example: usize,
        m: usize,
        l: usize,
        example_R1CS: ExampleR1CSLite<F>,
        cs_max_degree: usize,
        rng: &mut R,
    ) -> (
        (ProverPP<F>, VerifierPP<F>, C::CS_PP),
        (View<F>, View<F>, CView<P, C>),
    )
    where
        R: rand_core::RngCore,
    {
        // Setup of the parameters
        // For now, the PHP only take as input squared matrices of dimension a power of two
        // A function will come later to convert any PHP input to the appropriate size
        // We want n to be a power of two greater than n_example
        let n_next_pow2 = (3 * n_example).next_power_of_two();
        let m = 3 * n_example;

        assert!(
            n_next_pow2 <= n_max,
            "n value for example generation larger than n_max"
        );

        // Create new examples with random parameters
        // Inialization of the example
        let ExampleR1CSLite {
            matrices: (L, R),
            polynomials: (a_prime_poly, b_prime_poly),
            vector: x_vec,
        } = example_R1CS;
        let cs = CSLiteParams::new(n_next_pow2, n_next_pow2, l, L, R);

        // initialize parameters for prover
        let prover_PP = ProverPP {
            cs: cs.clone(),
            x_vec: x_vec.clone(),
            a_prime_poly,
            b_prime_poly,
        };

        // initialize parameters for verifier
        let verifier_PP = VerifierPP {
            cs: cs.clone(),
            x_vec: x_vec.clone(),
        };

        // Generate relation encoder values
        // Need to setup a way to ingrate them the the prover and verifier param
        let (n, m, l) = prover_PP.cs.get_params();
        let domain_k = GeneralEvaluationDomain::<F>::new(m).unwrap();
        let domain_k_size = domain_k.size();

        let mut oracles_vec = R1CSLite2x::setup(&prover_PP.cs);
        let v_crL = oracles_vec.remove(0).get_matrix().unwrap();
        let v_crR = oracles_vec.remove(0).get_matrix().unwrap();
        let cr = oracles_vec.remove(0).get_matrix().unwrap();

        // Parameters specific to R1CSLite2
        // imported from the prover.rs file
        const b_a: usize = 1;
        const b_b: usize = 2;
        const b_q: usize = 1;
        // const b_r: usize = 1;
        const b_s: usize = 1;

        // Setup of the CS parameters
        let D = cmp::max(
            2 * cs_max_degree + b_a + b_b + 2 * b_q - 3,
            cmp::max(cs_max_degree + b_s + b_q - 1, m),
        );

        let cs_pp: C::CS_PP = C::setup(rng, D);

        // Creating the view for the prover and the verifier
        // prover view has scalars and polynomials
        // verifer view has scalars
        let mut php_prover_view: View<F> = View::<F>::new();
        let php_verifier_view: View<F> = View::<F>::new();

        // Creation of the prover commitment view
        // Contains the useful commitments for the proofs and verifier checks
        let mut prover_Cview: CView<P, C> = CView::<P, C>::new();

        // Adds relation encoder polynomials to the prover view
        php_prover_view.add_idx_oracle(
            &stringify!(v_crL).to_string(),
            Oracle::OMtxPoly(v_crL.clone()),
        );
        php_prover_view.add_idx_oracle(
            &stringify!(v_crR).to_string(),
            Oracle::OMtxPoly(v_crR.clone()),
        );
        php_prover_view.add_idx_oracle(&stringify!(cr).to_string(), Oracle::OMtxPoly(cr.clone()));

        // Commit to relation encoder polynomials and add them to the commits view
        let commit_RE_polys = commit_oracles_dict_rel::<P, C>(&cs_pp, &php_prover_view.idx_oracles);
        prover_Cview.append_idx_comm_oracles(&commit_RE_polys);

        (
            (prover_PP, verifier_PP, cs_pp),
            (php_prover_view, php_verifier_view, prover_Cview),
        )
    }

    // proof done in two evaluations proofs
    pub fn prover(
        prover_PP: &ProverPP<F>,
        verifier_PP: &VerifierPP<F>,
        cs_pp: &C::CS_PP,
        php_p_view: &mut View<F>,
        php_v_view: &mut View<F>,
        c_p_view: &mut CView<P, C>,
    ) -> Result<(P::G1Projective, P::G1Projective), CPError>
    where
        C: CommitmentScheme<P, CS_PP = CS1_PP<P>> + CPEval<P, C>,
    {
        let prover_time = start_timer!(|| "Lunar_pover");

        let commit_RE_cr = c_p_view.get_idx_oracle_matrix("cr").unwrap();
        let commit_RE_vcrR = c_p_view.get_idx_oracle_matrix("v_crR").unwrap();
        let commit_RE_vcrL = c_p_view.get_idx_oracle_matrix("v_crL").unwrap();

        let mut vec_FS = Vec::new();
        for ((mut v1, mut v2), mut v3) in zip(zip(commit_RE_cr, commit_RE_vcrR), commit_RE_vcrL) {
            vec_FS.append(&mut v1);
            vec_FS.append(&mut v2);
            vec_FS.append(&mut v3);
        }
        let FS_commit_RE = <C as CommitmentScheme<P>>::extract_vec_commit_rel(vec_FS.clone())?;

        // Setup the FS RNG using the RE commitments
        let mut fs_rng = SimpleHashFiatShamirRng::<Blake2s, ChaChaRng>::initialize(
            &to_bytes![FS_commit_RE].unwrap(),
        );

        let x_vec = prover_PP.x_vec.clone();
        fs_rng.absorb(&to_bytes![x_vec].unwrap());

        R1CSLite2x::prover_rounds(
            prover_PP,
            verifier_PP,
            cs_pp,
            php_p_view,
            php_v_view,
            c_p_view,
            &mut fs_rng,
        )?;

        R1CSLite2x::prover_precomput(
            prover_PP,
            cs_pp,
            php_p_view,
            php_v_view,
            c_p_view,
            &mut fs_rng,
        )?;

        let proofs: (P::G1Projective, P::G1Projective) = R1CSLite2x::prover_proofs(
            prover_PP, cs_pp, php_p_view,
            // Only here to satisfy the requirement for P
            c_p_view,
        )?;

        end_timer!(prover_time);

        Ok(proofs)
    }

    // Check done in two pairings
    pub fn verifier(
        verifier_PP: &VerifierPP<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_view: &mut View<P::Fr>,
        prover_commits: &CView<P, C>,
        proofs: (P::G1Projective, P::G1Projective),
    ) -> Result<(), CPError>
    where
        C: CommitmentScheme<P, CS_PP = CS1_PP<P>> + CPEval<P, C>,
    {
        let now = Instant::now();

        let commit_RE_cr = prover_commits.get_idx_oracle_matrix("cr").unwrap();
        let commit_RE_vcrR = prover_commits.get_idx_oracle_matrix("v_crR").unwrap();
        let commit_RE_vcrL = prover_commits.get_idx_oracle_matrix("v_crL").unwrap();

        let mut vec_REL_commit = Vec::new();
        for ((mut v1, mut v2), mut v3) in zip(zip(commit_RE_cr, commit_RE_vcrR), commit_RE_vcrL) {
            vec_REL_commit.append(&mut v1);
            vec_REL_commit.append(&mut v2);
            vec_REL_commit.append(&mut v3);
        }
        let FS_commit_RE =
            <C as CommitmentScheme<P>>::extract_vec_commit_rel(vec_REL_commit.clone())?;

        // Setup the FS RNG using the RE commitments
        let mut fs_rng = SimpleHashFiatShamirRng::<Blake2s, ChaChaRng>::initialize(
            &to_bytes![FS_commit_RE].unwrap(),
        );

        let x_vec = verifier_PP.x_vec.clone();
        fs_rng.absorb(&to_bytes![x_vec].unwrap());

        let elapsed_time = now.elapsed();
        // println!("Verifier extracting RE commit took {} microsec.", elapsed_time.as_micros());
        let now = Instant::now();

        let mut php_v_view = R1CSLite2x::verifier_rounds(
            &verifier_PP,
            prover_view,
            cs_pp,
            &prover_commits,
            &mut fs_rng,
        )?;

        let elapsed_time = now.elapsed();
        // println!("Verifier ROUNDS took {} microsec.", elapsed_time.as_micros());
        let now = Instant::now();

        let ((c_p_prime_eq1, p_eq1_prime_at_y), (c_p_prime_eq2_degs, p_prime_eq2_degs_at_y2)) =
            R1CSLite2x::verifier_precompute(
                &verifier_PP,
                &mut php_v_view,
                prover_view,
                cs_pp,
                &prover_commits,
                &mut fs_rng,
            )?;

        let polys_evals = (p_eq1_prime_at_y, p_prime_eq2_degs_at_y2);

        let commits = (c_p_prime_eq1, c_p_prime_eq2_degs);

        let elapsed_time = now.elapsed();
        // println!("Verifier precomputation took {} microsec.", elapsed_time.as_micros());
        let now = Instant::now();

        let _ = R1CSLite2x::verifier_checks(
            &verifier_PP,
            &mut php_v_view,
            cs_pp,
            polys_evals,
            commits,
            proofs,
            &prover_commits,
        )?;

        let elapsed_time = now.elapsed();
        // println!("Verifier CHECKS took {} microsec.", elapsed_time.as_micros());
        let now = Instant::now();

        Ok(())
    }
}
