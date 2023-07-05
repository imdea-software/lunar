//#[cfg(feature = "std")]

use itertools::izip;
use itertools::Itertools;
use std::iter::zip;

use ark_std::fmt::Debug;

// Import Randomness
use ark_std::rand::Rng;
use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};
use rand::prelude::IteratorRandom;
use rand::thread_rng;

use ark_marlin::rng::FiatShamirRng;
use ark_marlin::rng::SimpleHashFiatShamirRng;
use ark_marlin::*;
use blake2::Blake2s;
use rand_chacha::ChaChaRng;

use ark_ff::{to_bytes, FftField, Field, One, PrimeField, Zero};
use ark_poly::{
    domain::EvaluationDomain,
    evaluations::univariate::Evaluations,
    multivariate::{SparsePolynomial as SPolynomial, SparseTerm, Term},
    polynomial::Polynomial,
    univariate::{DenseOrSparsePolynomial, DensePolynomial as DPolynomial},
    GeneralEvaluationDomain, MVPolynomial, UVPolynomial,
};

use dict::{Dict, DictIface};

use ark_bls12_381::Fr as bls12_381_fr;
use ark_bls12_381::{Bls12_381, Fr};
use ark_ec::{PairingEngine, ProjectiveCurve};

use crate::comms::*;
use crate::evalpolyproofs::*;
use crate::matrixutils::*;
use crate::php::*;
use crate::phplite2::*;
//use crate::matrixutils::{Matrix, lambda_x, lagrange_at_tau, masking_polynomial, phi, smatrix_rand, shift_by_n};

//#[macro_use]
//use crate::phplite2::view::add_to_views;
use crate::poly_const;
use crate::x;

use crate::add_to_views;

use crate::r1cslite::{example_simple, simple_matrixL, simple_matrixR};
use crate::r1cslite::{example_simple_v2, simple_matrixL_v2, simple_matrixR_v2, ExampleR1CSLite};

use std::cmp;
use std::ops::Add;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;

#[cfg(test)]
fn test_sparse_encoding<F: FftField, R: rand_core::RngCore>(
    dim: usize,
    density: usize,
    rng: &mut R,
) {
    let rand_smatrix = smatrix_rand::<F, R>(dim, density, rng);
    let sparse_encoding = rand_smatrix.encode_matrix();
    let _domain_k = GeneralEvaluationDomain::<F>::new(density).unwrap();
    let domain_h = GeneralEvaluationDomain::<F>::new(dim).unwrap();

    let row_evals = sparse_encoding.row_evals;
    let col_evals = sparse_encoding.col_evals;
    let val_evals = sparse_encoding.val_evals;

    for i in 0..density {
        let row = phi(&row_evals[i], domain_h).unwrap();
        let col = phi(&col_evals[i], domain_h).unwrap();
        let val = val_evals[i];
        // assert: (col, val) in rand_smatrix[row]
        if let Matrix::SparseMatrix(ref sparse) = rand_smatrix {
            assert!(sparse[row].iter().any(|&(c, v)| c == col && v == val));
        }
    }
}

#[cfg(test)]
fn single_poly_proof<P, C>(cs_degree: usize, deg_poly: usize)
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    let rng = &mut test_rng();
    let cs1_pp: CS1_PP<P> = CS1::setup(rng, cs_degree);

    //let domain_h = GeneralEvaluationDomain::<P::Fr>::new(domain_size).unwrap();

    let p = DPolynomial::<P::Fr>::rand(deg_poly, rng);

    // Test the check if the commitment/proof/verify works for a single polynomial
    let y = <P::Fr>::rand(rng);
    let b = p.evaluate(&y);
    let p_commit = CS1::commit_swh(&cs1_pp, &p);
    let proof = <CS1 as CPEval<P, C>>::prove_poly_eval(
        &cs1_pp,
        //&<CS1 as CommitmentScheme>::CommT, // not working, will be needed later for type checks
        &p,
        (&y, &b), // to be consistent with the notation of the paper (a,b)
    )
    .unwrap();
    let _verifier_check = <CS1 as CPEval<P, C>>::verify(
        &cs1_pp,
        //&<CS1 as CommitmentScheme>::CommT, // not working, will be needed later for type checks
        &proof,
        p_commit,
        (&y, &b), // to be consistent with the notation of the paper (a,b)
    )
    .unwrap();
}

#[cfg(test)]
fn single_poly_proof_CS2<P, C>(cs_degree: usize, deg_poly: usize)
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    let rng = &mut test_rng();
    let cs2_pp: CS2_PP<P> = CS2::setup(rng, cs_degree);

    //let domain_h = GeneralEvaluationDomain::<P::Fr>::new(domain_size).unwrap();

    let p = DPolynomial::<P::Fr>::rand(deg_poly, rng);

    // Test the check if the commitment/proof/verify works for a single polynomial
    let y = <P::Fr>::rand(rng);
    let b = p.evaluate(&y);
    let p_commit = CS2::commit_swh(&cs2_pp, &p);
    let proof = <CS2 as CPEval<P, C>>::prove_poly_eval(
        &cs2_pp,
        //&<CS1 as CommitmentScheme>::CommT, // not working, will be needed later for type checks
        &p,
        (&y, &b), // to be consistent with the notation of the paper (a,b)
    )
    .unwrap();
    let _verifier_check = <CS2 as CPEval<P, C>>::verify(
        &cs2_pp,
        //&<CS1 as CommitmentScheme>::CommT, // not working, will be needed later for type checks
        &proof,
        p_commit,
        (&y, &b), // to be consistent with the notation of the paper (a,b)
    )
    .unwrap();
}

// Can be complexify with more poly and different degree
#[cfg(test)]
fn simple_poly_proof_using_qeq<P, C>(cs_degree: usize, deg_poly: usize)
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    let rng = &mut test_rng();
    let cs2_pp: CS2_PP<P> = CS2::setup(rng, cs_degree);

    // Sample two polynomials
    let p1 = DPolynomial::<P::Fr>::from_coefficients_slice(&[P::Fr::one(), P::Fr::one()]); // X + 1
    let p2 = DPolynomial::<P::Fr>::from_coefficients_slice(&[P::Fr::one(), P::Fr::one()]); // X + 1

    // Generate the terms of the MV Poly
    let terms: Vec<(P::Fr, SparseTerm)> = vec![
        (P::Fr::one(), SparseTerm::new(vec![(0, 2)])),
        (-P::Fr::one(), SparseTerm::new(vec![(0, 1), (1, 1)])),
        (-P::Fr::one(), SparseTerm::new(vec![(0, 1)])),
        (P::Fr::one(), SparseTerm::new(vec![(1, 1)])),
    ];

    // generate the MV poly
    // number of variable for the MV poly, here = 2 (X and Y)
    let num_var = 2;
    let G: SPolynomial<P::Fr, SparseTerm> = SPolynomial::from_coefficients_vec(num_var, terms); // G(X,Y) = X**2 - X * Y - X + Y

    let (comm_pi_g1, comm_pi_g2) = <CS2 as CPqeq<P, C>>::compute_complementary_commits(
        &cs2_pp,
        vec![&p1, &p2],
        vec![&p1, &p2],
    );

    let check_2 =
        <CS2 as CPqeq<P, C>>::verify_complementary_commits(&cs2_pp, &comm_pi_g1, &comm_pi_g2)
            .unwrap();

    let result =
        <CS2 as CPqeq<P, C>>::verify_Ghat_eval(&cs2_pp, G, &comm_pi_g1, &comm_pi_g2).unwrap();
}

#[cfg(test)]
fn multiple_poly_proof_same_point<P, C>(cs_degree: usize, deg_poly: usize)
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    let rng = &mut test_rng();
    let cs1_pp: CS1_PP<P> = CS1::setup(rng, cs_degree);

    let mut pi = Vec::new();
    let mut bi = Vec::new();
    let mut ci = Vec::new();
    let mut rhoi = Vec::new();
    let mut poly;
    let mut eval;
    // random point to evaluate on
    let a = <P::Fr>::rand(rng);
    // create 10 random polynomials to be evaluated
    // compute the image of a for each poly
    // compute the commitments of each poly
    for _ in 0..10 {
        poly = DPolynomial::<P::Fr>::rand(deg_poly, rng);
        eval = poly.evaluate(&a);
        bi.push(eval.clone());
        pi.push(poly.clone());
        ci.push(CS1::commit_swh(&cs1_pp, &poly));
        rhoi.push(P::Fr::rand(rng).clone());
        println!("poly = {:?}\n", poly);
    }

    let (proof, b_star, c_star) = <CS1 as CPEvals<P, C>>::multiple_evals_on_same_point(
        &cs1_pp,
        //&<CS1 as CommitmentScheme>::CommT, // not working, will be needed later for type checks
        (a, bi),
        pi,
        ci,
        rhoi,
        //rng
    )
    .unwrap();

    let _verifier_check = <CS1 as CPEval<P, C>>::verify(
        &cs1_pp,
        //&<CS1 as CommitmentScheme>::CommT, // not working, will be needed later for type checks
        &proof,
        c_star,
        (&a, &b_star), // to be consistent with the notation of the paper (a,b)
    )
    .unwrap();
}

#[cfg(test)]
fn multiple_poly_proof_several_points<P, C>(cs_degree: usize, deg_poly: usize)
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    let rng = &mut test_rng();
    let cs1_pp: CS1_PP<P> = CS1::setup(rng, cs_degree);

    let mut poly;
    let mut eval;
    let mut a_vec = Vec::new();
    let mut b_vec = Vec::new();
    let mut p_vec = Vec::new();
    let mut c_vec = Vec::new();
    let mut rho_vec = Vec::new();

    for i in 0..3 {
        // generate random points on which to evaluate the polys
        a_vec.push(P::Fr::rand(rng).clone());

        let mut pi = Vec::new();
        let mut ci = Vec::new();
        let mut bi = Vec::new();
        let mut rhoi = Vec::new();

        // create 10 random polynomials to be evaluated
        // compute the image of a for each poly
        // compute the commitments of each poly
        for _ in 0..10 {
            poly = DPolynomial::<P::Fr>::rand(deg_poly, rng);
            eval = poly.evaluate(&a_vec[i]);
            bi.push(eval.clone());
            pi.push(poly.clone());
            let commit = CS1::commit_swh(&cs1_pp, &poly);
            /*            let commit = match CS1::commit_swh(&cs1_pp, &poly) {
            TypedComm::Rel(c) => c,
            TypedComm::Swh(c) => c,
            };*/
            ci.push(commit);
            rhoi.push(P::Fr::rand(rng).clone());
            println!("poly = {:?}\n", poly);
        }

        b_vec.push(bi);
        c_vec.push(ci);
        p_vec.push(pi);
        rho_vec.push(rhoi);
    }

    let proof_vec =
        <CS1 as CPEvals<P, C>>::multiple_evals(&cs1_pp, (&a_vec, b_vec), p_vec, c_vec, rho_vec)
            .unwrap();

    for (i, a) in a_vec.iter().enumerate() {
        let (proof, b_star, c_star) = proof_vec[i];

        let _verifier_check = <CS1 as CPEval<P, C>>::verify(
            &cs1_pp,
            &proof,
            c_star,
            (a, &b_star), // to be consistent with the notation of the paper (a,b)
        )
        .unwrap();
    }
}

#[cfg(test)]
fn proof_one_deg_bound_one_poly_CS2<P, C>(cs_degree: usize, deg_poly: usize)
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    let rng = &mut test_rng();
    let cs2_pp: CS2_PP<P> = CS2::setup(rng, cs_degree);

    let p = DPolynomial::<P::Fr>::rand(deg_poly, rng);

    let rho = P::Fr::rand(rng);

    let c = CS2::commit_swh(&cs2_pp, &p);

    let commit_c;
    match c {
        TypedComm::Swh(comm) => {
            commit_c = comm;
        }
        TypedComm::Rel(comm) => {
            commit_c = cs2_pp.g1;
        } // This case should never happen
    }

    let vk_d = <CS2 as CPdeg<P, C>>::derive(&cs2_pp, &vec![deg_poly]).unwrap();

    let (c_star, c_prime) = <CS2 as CPdeg<P, C>>::prove_one_deg_bound(
        &cs2_pp,
        deg_poly,
        vec![p],
        vec![commit_c],
        vec![rho],
    )
    .unwrap();

    let result = <CS2 as CPdeg<P, C>>::verify_one_deg_bound(&cs2_pp, vk_d[0], c_star, c_prime);
}

#[cfg(test)]
fn proof_one_deg_bound_several_polys_CS2<P, C>(cs_degree: usize, deg_poly: usize)
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    let rng = &mut test_rng();
    let cs2_pp: CS2_PP<P> = CS2::setup(rng, cs_degree);

    let mut pi = Vec::new();
    let mut ci = Vec::new();
    let mut rhoi = Vec::new();

    // create 10 random polynomials to be evaluated
    // compute the image of a for each poly
    // compute the commitments of each poly
    for _ in 0..10 {
        let poly = DPolynomial::<P::Fr>::rand(deg_poly, rng);
        pi.push(poly.clone());

        let comm_c = CS2::commit_swh(&cs2_pp, &poly);
        let temp_c;
        match comm_c {
            TypedComm::Swh(c) => {
                temp_c = c;
            }
            TypedComm::Rel(c) => {
                temp_c = cs2_pp.g1;
            } // This case should never happen
        }

        ci.push(temp_c);
        rhoi.push(P::Fr::rand(rng).clone());
        //println!("poly = {:?}\n", poly);
    }

    let vk_d = <CS2 as CPdeg<P, C>>::derive(&cs2_pp, &vec![deg_poly]).unwrap();

    let (c_star, c_prime) =
        <CS2 as CPdeg<P, C>>::prove_one_deg_bound(&cs2_pp, deg_poly, pi, ci, rhoi).unwrap();

    let result = <CS2 as CPdeg<P, C>>::verify_one_deg_bound(&cs2_pp, vk_d[0], c_star, c_prime);
}

#[cfg(test)]
fn proof_two_deg_bound_five_polys_CS2<P, C>(cs_degree: usize, max_deg: usize)
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    let rng = &mut test_rng();
    let cs2_pp: CS2_PP<P> = CS2::setup(rng, cs_degree);

    let degb1 = rng.gen_range(0..max_deg);
    let degb2 = rng.gen_range(0..max_deg);

    let deg_bounds = vec![degb1, degb2];

    let mut pi = Vec::<Vec<DPolynomial<P::Fr>>>::new();
    let mut ci = Vec::<Vec<_>>::new();
    let mut rhoi = Vec::<Vec<P::Fr>>::new();

    for (i, &d) in deg_bounds.iter().enumerate() {
        let mut polys = Vec::<DPolynomial<P::Fr>>::new();
        let mut ci_vec = Vec::<_>::new();
        let mut rhoi_vec = Vec::<P::Fr>::new();

        for _ in 0..5 {
            let poly = DPolynomial::<P::Fr>::rand(d, rng);
            //println!("poly = {:?}", poly);
            polys.push(poly.clone());

            let comm_c = CS2::commit_swh(&cs2_pp, &poly);
            // let temp_c;
            // match comm_c {
            //     TypedComm::Swh(c) => {temp_c = c;},
            //     TypedComm::Rel(c) => {temp_c = cs2_pp.g1;}, // This case should never happen
            // }

            ci_vec.push(comm_c);
            rhoi_vec.push(P::Fr::rand(rng).clone());
            //println!("poly = {:?}\n", poly);
        }
        pi.push(polys);
        ci.push(ci_vec);
        rhoi.push(rhoi_vec);
    }

    let vk_d = <CS2 as CPdeg<P, C>>::derive(&cs2_pp, &deg_bounds).unwrap();

    let (commit_star, commit_prime) =
        <CS2 as CPdeg<P, C>>::prove_deg_bound(&cs2_pp, &deg_bounds, pi, ci, rhoi).unwrap();

    let result = <CS2 as CPdeg<P, C>>::verify_deg_bound(
        &cs2_pp,
        &vk_d,
        &deg_bounds,
        commit_star,
        commit_prime,
    )
    .unwrap();
}

#[cfg(test)]
fn proof_anynum_deg_bound_anynum_polys_CS2<P, C>(
    cs_degree: usize,
    max_deg: usize,
    max_deg_bound: usize,
    max_poly: usize,
) where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    let rng = &mut test_rng();
    let cs2_pp: CS2_PP<P> = CS2::setup(rng, cs_degree);

    let num_deg_bound = rng.gen_range(0..max_deg_bound);
    let mut deg_bounds = Vec::<_>::new();
    let mut num_polys = Vec::<_>::new();

    for _ in 0..num_deg_bound {
        let degb1 = rng.gen_range(0..max_deg);
        let num_poly = rng.gen_range(0..max_poly);
        deg_bounds.push(degb1);
        num_polys.push(num_poly);
    }

    let mut pi = Vec::<Vec<DPolynomial<P::Fr>>>::new();
    let mut ci = Vec::<Vec<_>>::new();
    let mut rhoi = Vec::<Vec<P::Fr>>::new();

    for (i, &d) in deg_bounds.iter().enumerate() {
        let mut polys = Vec::<DPolynomial<P::Fr>>::new();
        let mut ci_vec = Vec::<_>::new();
        let mut rhoi_vec = Vec::<P::Fr>::new();

        for _ in 0..num_polys[i] {
            let poly = DPolynomial::<P::Fr>::rand(d, rng);
            //println!("poly = {:?}", poly);
            polys.push(poly.clone());

            let comm_c = CS2::commit_swh(&cs2_pp, &poly);
            // let temp_c;
            // match comm_c {
            //     TypedComm::Swh(c) => {temp_c = c;},
            //     TypedComm::Rel(c) => {temp_c = cs2_pp.g1;}, // This case should never happen
            // }

            ci_vec.push(comm_c);
            rhoi_vec.push(P::Fr::rand(rng).clone());
            //println!("poly = {:?}\n", poly);
        }
        pi.push(polys);
        ci.push(ci_vec);
        rhoi.push(rhoi_vec);
    }

    let vk_d = <CS2 as CPdeg<P, C>>::derive(&cs2_pp, &deg_bounds).unwrap();

    let (commit_star, commit_prime) =
        <CS2 as CPdeg<P, C>>::prove_deg_bound(&cs2_pp, &deg_bounds, pi, ci, rhoi).unwrap();

    let result = <CS2 as CPdeg<P, C>>::verify_deg_bound(
        &cs2_pp,
        &vk_d,
        &deg_bounds,
        commit_star,
        commit_prime,
    )
    .unwrap();
}

//
// COMPOSED FUNCTIONS
//

// PHPlite2x v2 with batch proofs
fn phplite2x_batch_proof_verif<P, F, C>(
    n_max: usize,
    n_example: usize,
    m: usize,
    l: usize,
    cs_max_degree: usize,
    //mut rng: R
) where
    P: PairingEngine + PairingEngine<Fr = F>,
    F: FftField,
    C: CommitmentScheme<P, CS_PP = CS1_PP<P>> + CPEval<P, C>, // Find a way to remove CS_PP condition
                                                              //RNG: rand_core::RngCore
{
    let mut rng = thread_rng();

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
    } = example_simple_v2::<F, _>(n_example, n_next_pow2, l, &mut rng);
    let cs = CSLiteParams::new(n_next_pow2, n_next_pow2, l, L, R);

    //let k_domain_size = m;
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
    let domain_k = GeneralEvaluationDomain::<P::Fr>::new(m).unwrap();
    let domain_k_size = domain_k.size();

    let mut oracles_vec = R1CSLite2x::setup(&prover_PP.cs);
    let v_crL = oracles_vec.remove(0).get_matrix().unwrap();
    let v_crR = oracles_vec.remove(0).get_matrix().unwrap();
    let cr = oracles_vec.remove(0).get_matrix().unwrap();

    /// Parameters specific to R1CSLite2
    // imported from the prover.rs file
    // Are those parameters fixed?
    const b_a: usize = 1;
    const b_b: usize = 2;
    const b_q: usize = 1;
    // const b_r: usize = 1;
    const b_s: usize = 1;

    // Setup of the CS parameters // change with max values for n and m
    //let D = cmp::max(2 * n + b_a + b_b + 2 * b_q - 3, cmp::max(n + b_s + b_q - 1, m));
    let D = cmp::max(
        2 * cs_max_degree + b_a + b_b + 2 * b_q - 3,
        cmp::max(cs_max_degree + b_s + b_q - 1, m),
    );
    println!("D = {:?}", D);
    let cs_pp: C::CS_PP = C::setup(&mut rng, D);

    // Creation of prover and verifier view
    // prover view has scalars and polynomials
    // verifer view has scalars
    let mut php_prover_view: View<P::Fr> = View::<P::Fr>::new();
    let mut php_verifier_view: View<P::Fr> = View::<P::Fr>::new();

    // Creation of the prover commitment view
    // Contains the useful commitments for the proofs and verifier checks
    // TODO: Check that there are not useless commits for the verifier
    // if so create a separate view of the verifier with less commits
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

    println!("RE DONE");
    println!("Starting Proof");

    let proof_vec = prover_phplite2x_batched::<P, C>(
        &prover_PP,
        &verifier_PP, // remove it
        &mut php_prover_view,
        &mut php_verifier_view,
        &mut prover_Cview,
        &cs_pp,
    )
    .unwrap();

    println!("Proof DONE");
    println!("Starting Verification");

    verifier_phplite2x_batched::<P, C>(
        verifier_PP,
        &cs_pp,
        &mut php_verifier_view,
        prover_Cview,
        proof_vec,
    )
    .unwrap();

    println!("Verification DONE");
}

// function that simulates the prover execution and computes the proof
fn prover_phplite2x_batched<P, C>(
    prover_PP: &ProverPP<P::Fr>,
    verifier_PP: &VerifierPP<P::Fr>,
    php_p_view: &mut View<P::Fr>,
    php_v_view: &mut View<P::Fr>,
    c_p_view: &mut CView<P, C>,
    cs_pp: &C::CS_PP,
) -> Result<(P::G1Projective, P::G1Projective), CPError>
where
    P: PairingEngine, // + PairingEngine<Fr = F>,
    C: CommitmentScheme<P, CS_PP = CS1_PP<P>> + CPEval<P, C>,
{
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

    R1CSLite2x::prover_rounds(
        prover_PP,
        verifier_PP,
        cs_pp,
        php_p_view,
        php_v_view,
        c_p_view,
        &mut fs_rng,
    );

    R1CSLite2x::prover_precomput(
        prover_PP,
        cs_pp,
        php_p_view,
        php_v_view,
        c_p_view,
        &mut fs_rng,
    );

    let proofs: (P::G1Projective, P::G1Projective) = R1CSLite2x::prover_proofs(
        prover_PP, cs_pp, php_p_view,
        // Only here to satisfy the requirement for P
        c_p_view, // To be removed
    )?;

    Ok(proofs)
}

fn verifier_phplite2x_batched<P, C>(
    verifier_PP: VerifierPP<P::Fr>,
    cs_pp: &C::CS_PP,
    prover_view: &mut View<P::Fr>,
    prover_commits: CView<P, C>,
    proofs: (P::G1Projective, P::G1Projective),
) -> Result<(), CPError>
where
    P: PairingEngine, // + PairingEngine<Fr = F>,
    C: CommitmentScheme<P, CS_PP = CS1_PP<P>> + CPEval<P, C>,
{
    let commit_RE_cr = prover_commits.get_idx_oracle_matrix("cr").unwrap();
    let commit_RE_vcrR = prover_commits.get_idx_oracle_matrix("v_crR").unwrap();
    let commit_RE_vcrL = prover_commits.get_idx_oracle_matrix("v_crL").unwrap();

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

    // Create the verifier view

    let mut php_v_view = R1CSLite2x::verifier_rounds(
        &verifier_PP,
        prover_view,
        cs_pp,
        &prover_commits,
        &mut fs_rng,
    )?;

    let alpha = php_v_view.get_scalar("alpha").unwrap();
    let x = php_v_view.get_scalar("x").unwrap();
    let y = php_v_view.get_scalar("y").unwrap();

    // let (c_p_prime_eq1, p_eq1_prime_at_y) = verifier_phplite2_precompute_eq1::<P, C, _>(
    //     &verifier_PP,
    //     &mut php_v_view,
    //     prover_view,
    //     cs_pp,
    //     &prover_commits,
    //     &mut fs_rng,
    // );

    // let (c_p_prime_eq2_degs, p_prime_eq2_degs_at_y2) =
    //     verifier_phplite2x_precompute_eq2_deg_checks::<P, C, _>(
    //         &verifier_PP,
    //         &mut php_v_view,
    //         prover_view,
    //         cs_pp,
    //         &prover_commits,
    //         &mut fs_rng,
    //     );

    let ((c_p_prime_eq1, p_eq1_prime_at_y), (c_p_prime_eq2_degs, p_prime_eq2_degs_at_y2)) =
        R1CSLite2x::verifier_precompute(
            &verifier_PP,
            &mut php_v_view,
            prover_view,
            cs_pp,
            &prover_commits,
            &mut fs_rng,
        )?;

    // let vec_polys_evals = vec![p_eq1_prime_at_y, p_prime_eq2_degs_at_y2];
    let polys_evals = (p_eq1_prime_at_y, p_prime_eq2_degs_at_y2);

    // let commits = (c_p_prime_eq1, c_p_prime_eq2_degs);
    let commits = (c_p_prime_eq1, c_p_prime_eq2_degs);

    println!("BEGINNING BATCH CHECK with two pairings total\n");

    let _ = R1CSLite2x::verifier_checks(
        &verifier_PP,
        &mut php_v_view,
        cs_pp,
        polys_evals,
        commits,
        proofs,
        &prover_commits,
    )
    .unwrap();

    println!("END BATCH CHECK\n");

    Ok(())
}

pub fn full_phplite2_prover_verifier_composed<P, F, C>(
    n_max: usize,
    n_example: usize,
    m: usize,
    l: usize,
    cs_max_degree: usize,
    //mut rng: R
) where
    P: PairingEngine + PairingEngine<Fr = F>,
    F: FftField,
    C: CommitmentScheme<P, CS_PP = CS2_PP<P>> + CPEval<P, C> + CPdeg<P, C> + CPqeq<P, C>, // Find a way to remove CS_PP condition
                                                                                          //RNG: rand_core::RngCore
{
    // let mut rng = thread_rng();
    let mut rng = &mut ark_std::test_rng();

    println!("first rng = {}", <P as PairingEngine>::Fr::rand(&mut rng));

    // Setup of the parameters
    // For now, the PHP only take as input squared matrices of dimension a power of two
    // A function will come later to convert any PHP input to the appropriate size
    // We want n to be a power of two greater than n_example
    let n_next_pow2 = (3 * n_example).next_power_of_two();
    let m = 3 * n_example;

    // Create new examples with random parameters
    // Inialization of the example
    let ExampleR1CSLite {
        matrices: (L, R),
        polynomials: (a_prime_poly, b_prime_poly),
        vector: x_vec,
    } = example_simple_v2::<F, _>(n_example, n_next_pow2, l, &mut rng);

    let cs = CSLiteParams::new(n_next_pow2, n_next_pow2, l, L, R);
    //let oracles_vec = R1CSLite2::setup(&cs);

    //let k_domain_size = m;
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
    let domain_k = GeneralEvaluationDomain::<P::Fr>::new(m).unwrap();
    let domain_k_size = domain_k.size();

    let mut oracles_vec = R1CSLite2::setup(&prover_PP.cs);
    let v_crL = oracles_vec.remove(0).get_matrix().unwrap();
    let v_crR = oracles_vec.remove(0).get_matrix().unwrap();
    let cr = oracles_vec.remove(0).get_matrix().unwrap();
    let cr_p = oracles_vec.remove(0).get_matrix().unwrap();

    /// Parameters specific to R1CSLite2
    // imported from the prover.rs file
    // Are those parameters fixed?
    const b_a: usize = 1;
    const b_b: usize = 2;
    const b_q: usize = 1;
    // const b_r: usize = 1;
    const b_s: usize = 1;

    // Setup of the CS parameters // change with max values for n and m
    //let D = cmp::max(2 * n + b_a + b_b + 2 * b_q - 3, cmp::max(n + b_s + b_q - 1, m));
    let D = cmp::max(
        2 * cs_max_degree + b_a + b_b + 2 * b_q - 3,
        cmp::max(cs_max_degree + b_s + b_q - 1, m),
    );
    // println!("D = {:?}", D);
    let cs_pp: C::CS_PP = C::setup(&mut rng, D);

    // Creation of prover and verifier view
    // prover view has scalars and polynomials
    // verifer view has scalars
    let mut php_prover_view: View<P::Fr> = View::<P::Fr>::new();
    let mut php_verifier_view: View<P::Fr> = View::<P::Fr>::new();

    // Creation of the prover commitment view
    // Contains the useful commitments for the proofs and verifier checks
    // TODO: Check that there are not useless commits for the verifier
    // if so create a separate view of the verifier with less commits
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

    // Degree vector for the degree check proof
    // Is there more check to do?
    let deg_vec = vec![n - 2, domain_k_size - 2];

    let vk_d = <C as CPdeg<P, C>>::derive(&cs_pp, &deg_vec).unwrap();

    println!("RE DONE");
    println!("Starting Proof");

    let proof = prover_actions_phplite2_composed::<P, C>(
        &prover_PP,
        &verifier_PP,
        &mut php_prover_view,
        &mut php_verifier_view,
        &mut prover_Cview,
        &cs_pp,
    )
    .unwrap();

    println!("Proof DONE");
    println!("Starting Verification");

    verifier_actions_phplite2_composed::<P, C>(
        verifier_PP,
        &cs_pp,
        prover_Cview,
        proof,
        &mut php_verifier_view, // for transfering the scalars => Change for the v view?
        vk_d,
    )
    .unwrap();

    println!("Verification DONE");
}

fn prover_actions_phplite2_composed<P, C>(
    prover_PP: &ProverPP<P::Fr>,
    verifier_PP: &VerifierPP<P::Fr>,
    php_p_view: &mut View<P::Fr>,
    php_v_view: &mut View<P::Fr>,
    c_p_view: &mut CView<P, C>,
    cs_pp: &C::CS_PP,
) -> Result<P::G1Projective, CPError>
where
    P: PairingEngine, // + PairingEngine<Fr = F>,
    C: CommitmentScheme<P, CS_PP = CS2_PP<P>> + CPEval<P, C> + CPdeg<P, C> + CPqeq<P, C>,
    // R: rand_core::RngCore
{
    let commit_RE_cr = c_p_view.get_idx_oracle_matrix("cr").unwrap();
    let commit_RE_cr_p = c_p_view.get_idx_oracle_matrix("cr_p").unwrap();
    let commit_RE_vcrL = c_p_view.get_idx_oracle_matrix("v_crL").unwrap();
    let commit_RE_vcrR = c_p_view.get_idx_oracle_matrix("v_crR").unwrap();

    let mut flattened_commit_RE_cr: Vec<C::CommT> = commit_RE_cr.into_iter().flatten().collect();
    let mut flattened_commit_RE_cr_p: Vec<C::CommT> =
        commit_RE_cr_p.into_iter().flatten().collect();
    let mut flattened_commit_RE_vcrL: Vec<C::CommT> =
        commit_RE_vcrL.into_iter().flatten().collect();
    let mut flattened_commit_RE_vcrR: Vec<C::CommT> =
        commit_RE_vcrR.into_iter().flatten().collect();

    let mut vec_REL_commit = Vec::new();
    vec_REL_commit.append(&mut flattened_commit_RE_cr);
    vec_REL_commit.append(&mut flattened_commit_RE_cr_p);
    vec_REL_commit.append(&mut flattened_commit_RE_vcrL);
    vec_REL_commit.append(&mut flattened_commit_RE_vcrR);

    let FS_commit_RE = <C as CommitmentScheme<P>>::extract_vec_commit_rel(vec_REL_commit.clone())?;

    // Setup the FS RNG using the RE commitments
    let mut fs_rng = SimpleHashFiatShamirRng::<Blake2s, ChaChaRng>::initialize(
        &to_bytes![FS_commit_RE].unwrap(),
    );

    R1CSLite2::prover_rounds(
        prover_PP,
        verifier_PP,
        cs_pp,
        php_p_view,
        php_v_view,
        c_p_view,
        &mut fs_rng,
    );

    R1CSLite2::prover_precomput(
        prover_PP,
        cs_pp,
        php_p_view,
        php_v_view,
        c_p_view,
        &mut fs_rng,
    );

    let proof = R1CSLite2::prover_proofs(prover_PP, cs_pp, php_p_view, c_p_view)?;

    Ok(proof)
}

fn verifier_actions_phplite2_composed<P, C>(
    verifier_PP: VerifierPP<P::Fr>,
    cs_pp: &C::CS_PP,
    prover_commits: CView<P, C>,
    proof: P::G1Projective,
    prover_view: &mut View<P::Fr>,
    vk_d: Vec<P::G2Projective>,
) -> Result<(), CPError>
where
    P: PairingEngine, // + PairingEngine<Fr = F>,
    C: CommitmentScheme<P, CS_PP = CS2_PP<P>> + CPEval<P, C> + CPdeg<P, C> + CPqeq<P, C>,
{
    let commit_RE_cr = prover_commits.get_idx_oracle_matrix("cr").unwrap();
    let commit_RE_cr_p = prover_commits.get_idx_oracle_matrix("cr_p").unwrap();
    let commit_RE_vcrL = prover_commits.get_idx_oracle_matrix("v_crL").unwrap();
    let commit_RE_vcrR = prover_commits.get_idx_oracle_matrix("v_crR").unwrap();

    // for c in commit_RE_cr.iter() {
    //     println!("\n{:?}\n", c);
    // }

    let flattened_commit_RE_cr: Vec<C::CommT> = commit_RE_cr.into_iter().flatten().collect();
    let flattened_commit_RE_cr_p: Vec<C::CommT> = commit_RE_cr_p.into_iter().flatten().collect();
    let flattened_commit_RE_vcrL: Vec<C::CommT> = commit_RE_vcrL.into_iter().flatten().collect();
    let flattened_commit_RE_vcrR: Vec<C::CommT> = commit_RE_vcrR.into_iter().flatten().collect();

    // for i in 0..3 {
    //     println!("{:?}", flattened_commit_RE_cr[i]);
    // }

    let mut vec_REL_commit = Vec::new();
    vec_REL_commit.append(&mut flattened_commit_RE_cr.clone());
    vec_REL_commit.append(&mut flattened_commit_RE_cr_p.clone());
    vec_REL_commit.append(&mut flattened_commit_RE_vcrL.clone());
    vec_REL_commit.append(&mut flattened_commit_RE_vcrR.clone());

    let FS_commit_RE = <C as CommitmentScheme<P>>::extract_vec_commit_rel(vec_REL_commit.clone())?;

    // Setup the FS RNG using the RE commitments
    let mut fs_rng = SimpleHashFiatShamirRng::<Blake2s, ChaChaRng>::initialize(
        &to_bytes![FS_commit_RE].unwrap(),
    );

    // println!("random elem = {:?}", P::Fr::rand(&mut fs_rng));

    // Create the verifier view
    let mut php_v_view = R1CSLite2::verifier_rounds(
        &verifier_PP,
        prover_view,
        cs_pp,
        &prover_commits,
        &mut fs_rng,
    )?;
    // println!("random elem = {:?}", P::Fr::rand(&mut fs_rng));

    let ((c_p_prime_eq1, p_eq1_prime_at_y), (cr_coeff, cr_prime_coeff, vcrL_vcrR_coeff)) =
        R1CSLite2::verifier_precompute_for_batched(
            &verifier_PP,
            &mut php_v_view,
            prover_view,
            cs_pp,
            &prover_commits,
            &mut fs_rng,
        );

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
    )
    .unwrap();

    Ok(())
}

///  Generation of a square sparse matrix with elements in the field *field*
/// smatrix_rand::<*field*>(*dimension*, *non_zeros*, *rng*)
#[test]
fn sample_sparse_matrix() {
    let rng = &mut test_rng();
    let rand_smatrix = smatrix_rand::<bls12_381_fr, _>(5, 10, rng);
    assert_eq!(rand_smatrix.count_non_zeros(), 10);
}

/// phi function maps the domain of size n into [n] canonically
/// phi(K[i]) == i
#[test]
fn test_phi() {
    let d_size: usize = 6;
    let domain = GeneralEvaluationDomain::<bls12_381_fr>::new(d_size).unwrap();
    let vec_domain: Vec<bls12_381_fr> = domain.elements().collect();
    for i in 0..d_size {
        assert_eq!(phi(&vec_domain[i], domain), Some(i));
    }
}

///// testing the inverse of phi
///// K[phi[K[i]]] == K[i]
#[test]
fn test_phi_inverse() {
    let d_size: usize = 6;
    let domain = GeneralEvaluationDomain::<bls12_381_fr>::new(d_size).unwrap();
    let vec_domain: Vec<bls12_381_fr> = domain.elements().collect();
    for i in 0..d_size {
        assert_eq!(
            vec_domain[phi(&vec_domain[i], domain).unwrap()],
            vec_domain[i]
        );
    }
}

/// Testing the correctness of the encoding of a sparse matrix
#[test]
fn encode_sparse_matrix() {
    let mut rng = test_rng();
    test_sparse_encoding::<bls12_381_fr, _>(4, 6, &mut rng);
}

/// Testing if the masking polynomial agrees with the masked poly in the given subset
#[test]
fn test_masking_polynomial() {
    let mut rng = test_rng();
    let domain_size = 16;
    let _subset_size = 4;
    let b = 2;
    let deg_p = 5;
    let p = DPolynomial::<bls12_381_fr>::rand(deg_p, &mut rng);
    let domain = GeneralEvaluationDomain::<bls12_381_fr>::new(domain_size).unwrap();
    let subset = domain.elements().choose_multiple(&mut rng, 4);
    let p_masked = masking_polynomial(&p, b, &subset, &mut rng);
    // p_masked agrees with p on subset:
    assert!(subset
        .iter()
        .all(|eta| p.evaluate(&eta) == p_masked.evaluate(&eta)));
}

/// test of the lagrange bivariate polynomial in the variable x
/// test on the degree and the property that ΛH(X, η) = LH,η(X)
#[test]
fn test_lambda() {
    let mut rng = test_rng();
    let domain_size = 16;
    let domain = GeneralEvaluationDomain::<bls12_381_fr>::new(domain_size).unwrap();
    let x = bls12_381_fr::rand(&mut rng);
    let lambda_x = lambda_x(x, domain);
    assert!(lambda_x.degree() == domain_size - 1);
    domain.elements().for_each(|tau| {
        assert_eq!(
            lambda_x.evaluate(&tau),
            lagrange_at_tau(domain, tau).evaluate(&x)
        )
    });
}

/// Testing the correctness of the proof and verification
#[test]
fn test_single_poly_proof_CS1() {
    single_poly_proof::<Bls12_381, CS1>(32, 3);
}

/// Testing the correctness of the proof and verification
#[test]
fn test_single_poly_proof_CS2() {
    single_poly_proof_CS2::<Bls12_381, CS2>(32, 3);
}

/// Testing the correctness of the proof and verification
#[test]
fn test_multi_poly_proof_single_point() {
    multiple_poly_proof_same_point::<Bls12_381, CS1>(32, 10);
}

/// Testing the correctness of the proof and verification
#[test]
fn test_multiple_poly_proof_several_points() {
    multiple_poly_proof_several_points::<Bls12_381, CS1>(32, 10);
}

/// Testing the correctness of the proof and verification
#[test]
fn test_simple_poly_proof_CPqeq() {
    simple_poly_proof_using_qeq::<Bls12_381, CS1>(32, 10);
}

/// Testing the correctness of the proof and verification
#[test]
fn test_proof_one_deg_bound_one_poly_CS2() {
    proof_one_deg_bound_one_poly_CS2::<Bls12_381, CS2>(10, 9);
}

/// Testing the correctness of the proof and verification
#[test]
fn test_proof_one_deg_bound_several_polys_CS2() {
    proof_one_deg_bound_several_polys_CS2::<Bls12_381, CS2>(10, 9);
}

/// Testing the correctness of the proof and verification
#[test]
fn test_proof_two_deg_bound_one_poly_CS2() {
    proof_two_deg_bound_five_polys_CS2::<Bls12_381, CS2>(10, 4);
}

/// Testing the correctness of the proof and verification
#[test]
fn test_proof_anynum_deg_bound_anynum_polys_CS2() {
    proof_anynum_deg_bound_anynum_polys_CS2::<Bls12_381, CS2>(32, 20, 10, 10);
}

/// Testing the correctness of the proof and verification
#[test]
fn test_phplite2_prover_batch_verif_verifier_composed() {
    // params: n_max, n_example, m, l, cs_max_deg
    full_phplite2_prover_verifier_composed::<Bls12_381, bls12_381_fr, CS2>(32, 10, 6, 7, 64);
    // assert!(false);
    // full_phplite2x_prover_verifier_batch_proof_verif_composed::<Bls12_381, bls12_381_fr, CS1>(128, 40, 24, 28, 512);
    // full_phplite2x_prover_verifier_batch_proof_verif_composed::<Bls12_381, bls12_381_fr, CS1>(256, 80, 48, 56, 1024);

    // Add test ffo false proof
}

/// Testing the correctness of the proof and verification
#[test]
fn test_phplite2x_prover_two_evals_batch_verif() {
    phplite2x_batch_proof_verif::<Bls12_381, bls12_381_fr, CS1>(32, 10, 6, 7, 64);
}
