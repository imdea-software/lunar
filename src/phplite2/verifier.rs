use crate::php::*;
use std::marker::PhantomData;

// Import Randomness
use ark_std::rand::Rng;
use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};
use rand::prelude::IteratorRandom;
use rand::thread_rng;

use ark_marlin::rng::FiatShamirRng;
use ark_marlin::rng::SimpleHashFiatShamirRng;

use ark_bls12_381::Fr as bls12_381_fr;
use ark_bls12_381::{Bls12_381, Fr};
use ark_ec::{PairingEngine, ProjectiveCurve};
use ark_ff::{to_bytes, FftField, Field, One, PrimeField, Zero};

use ark_poly::{
    domain::EvaluationDomain,
    evaluations::univariate::Evaluations,
    multivariate::{SparsePolynomial as SPolynomial, SparseTerm, Term},
    polynomial::Polynomial,
    univariate::{DenseOrSparsePolynomial, DensePolynomial as DPolynomial},
    GeneralEvaluationDomain, MVPolynomial, UVPolynomial,
};

use super::view::*;

use crate::comms::*;
use crate::evalpolyproofs::*;
use crate::matrixutils::*;

use crate::phplite2::{R1CSLite2, R1CSLite2x};

use crate::phplite2::*;

use crate::poly_const;
use crate::x;

#[derive(Clone)]
pub struct VerifierPP<F: FftField> {
    pub cs: CSLiteParams<F>,
    pub x_vec: Vec<F>,
}

pub trait VerifierR1CSLite2<P, F, C>
where
    P: PairingEngine + PairingEngine<Fr = F>,
    F: FftField,
    C: CommitmentScheme<P, CS_PP = CS2_PP<P>> + CPEval<P, C> + CPdeg<P, C> + CPqeq<P, C>,
{
    fn v_next_msg<R: rand_core::RngCore + FiatShamirRng>(
        rnd_idx: usize,
        verifier_input: &VerifierPP<F>,
        v_view: &mut View<F>,
        rng: &mut R,
    ) -> View<F>;

    fn verifier_rounds<R: rand_core::RngCore + FiatShamirRng>(
        verifier_PP: &VerifierPP<P::Fr>,
        p_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_commits: &CView<P, C>,
        fs_rng: &mut R,
    ) -> Result<View<P::Fr>, CPError>;

    fn verifier_precompute<R: rand_core::RngCore + FiatShamirRng>(
        verifier_PP: &VerifierPP<P::Fr>,
        v_view: &mut View<P::Fr>,
        p_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_commits: &CView<P, C>,
        fs_rng: &mut R,
    ) -> (SPolynomial<P::Fr, SparseTerm>, C::CommT, P::Fr);

    fn verifier_checks(
        verifier_PP: &VerifierPP<P::Fr>,
        cs_pp: &C::CS_PP,
        v_view: &mut View<P::Fr>,
        prover_commits: &CView<P, C>,
        vec_REL_commit: Vec<C::CommT>,
        vk_d: Vec<P::G2Projective>,
        prover_proofs: Vec<P::G1Projective>,
        G: SPolynomial<P::Fr, SparseTerm>,
        c_p_prime: C::CommT,
        p_prime_at_y: P::Fr,
    ) -> Result<(), CPError>;

    fn verifier_precompute_for_batched<R: rand_core::RngCore + FiatShamirRng>(
        verifier_PP: &VerifierPP<P::Fr>,
        v_view: &mut View<P::Fr>,
        p_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_commits: &CView<P, C>,
        fs_rng: &mut R,
    ) -> ((C::CommT, P::Fr), (Vec<P::Fr>, Vec<P::Fr>, Vec<P::Fr>));

    fn verifier_checks_batched(
        verifier_PP: &VerifierPP<P::Fr>,
        cs_pp: &C::CS_PP,
        v_view: &mut View<P::Fr>,
        prover_commits: &CView<P, C>,
        commit_RE_cr: Vec<C::CommT>,
        commit_RE_cr_p: Vec<C::CommT>,
        commit_RE_vcrL: Vec<C::CommT>,
        commit_RE_vcrR: Vec<C::CommT>,
        cr_coeff: Vec<P::Fr>,
        cr_prime_coeff: Vec<P::Fr>,
        vcrL_vcrR_coeff: Vec<P::Fr>,
        vk_d: Vec<P::G2Projective>,
        prover_proof: P::G1Projective,
        c_p_prime: C::CommT,
        p_prime_at_y: P::Fr,
    ) -> Result<(), CPError>;
}

pub trait VerifierR1CSLite2x<P, F, C>
where
    P: PairingEngine + PairingEngine<Fr = F>,
    F: FftField,
    C: CommitmentScheme<P, CS_PP = CS1_PP<P>> + CPEval<P, C>,
{
    fn v_next_msg<R: rand_core::RngCore + FiatShamirRng>(
        rnd_idx: usize,
        verifier_input: &VerifierPP<F>,
        v_view: &mut View<F>,
        rng: &mut R,
    ) -> View<F>;

    fn verifier_rounds<R: rand_core::RngCore + FiatShamirRng>(
        verifier_PP: &VerifierPP<P::Fr>,
        p_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_commits: &CView<P, C>,
        fs_rng: &mut R,
    ) ->  Result<View<P::Fr>, CPError>;

    fn verifier_precompute<R: rand_core::RngCore + FiatShamirRng>(
        verifier_PP: &VerifierPP<P::Fr>,
        v_view: &mut View<P::Fr>,
        p_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_commits: &CView<P, C>,
        fs_rng: &mut R,
    ) -> Result<((C::CommT, P::Fr), (C::CommT, P::Fr)), CPError>;

    fn verifier_checks(
        verifier_PP: &VerifierPP<P::Fr>,
        v_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        polys_eval: (P::Fr, P::Fr),
        commits_polys: (C::CommT, C::CommT),
        prover_proofs: (P::G1Projective, P::G1Projective),
        _phantom_view: &CView<P, C>,
    ) -> Result<(), CPError>;
}

impl<P, F, C> VerifierR1CSLite2<P, F, C> for R1CSLite2
where
    P: PairingEngine + PairingEngine<Fr = F>,
    F: FftField,
    C: CommitmentScheme<P, CS_PP = CS2_PP<P>> + CPEval<P, C> + CPdeg<P, C> + CPqeq<P, C>,
    // R: rand_core::RngCore + FiatShamirRng,
{
    fn v_next_msg<R: rand_core::RngCore + FiatShamirRng>(
        rnd_idx: usize,
        verifier_input: &VerifierPP<F>,
        v_view: &mut View<F>,
        rng: &mut R,
    ) -> View<F> {
        let mut p_view: View<F> = View::<F>::new();

        /* Common boilerplate code */
        let domain_h = GeneralEvaluationDomain::<F>::new(verifier_input.cs.get_params().0).unwrap();

        match rnd_idx {
            1 => {
                // Check the rng rand function
                let alpha = <F>::rand(rng);
                let x = <F>::rand(rng);

                add_to_views!(scalar, p_view, v_view, alpha);
                add_to_views!(scalar, p_view, v_view, x);
            }
            2 => {
                let y = domain_h.sample_element_outside_domain(rng);

                add_to_views!(scalar, p_view, v_view, y);
            }
            3 => {

                // do nothing
            }

            _ => {
                unimplemented!()
            }
        }

        p_view
    }

    fn verifier_rounds<R: rand_core::RngCore + FiatShamirRng>(
        verifier_PP: &VerifierPP<P::Fr>,
        p_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_commits: &CView<P, C>,
        fs_rng: &mut R,
    ) -> Result<View<P::Fr>, CPError> {
        // Useless view for now. Check if we need it at some point
        let mut php_v_view: View<P::Fr> = View::<P::Fr>::new();

        //--------------------- ROUND 1 ---------------------//

        // recover the commits of a_hat_prime and b_hat_prime and s
        let c_a_hat_prime = prover_commits.get_prv_oracle("a_hat_prime").unwrap();
        let c_b_hat_prime = prover_commits.get_prv_oracle("b_hat_prime").unwrap();
        let c_s = prover_commits.get_prv_oracle("s").unwrap();

        // Find a better way to absorb the commits?
        // ABSORBING THE FIRST COMMITMENTS
        let vec_comm_to_byte = C::extract_vec_commit_swh(vec![
            c_a_hat_prime.clone(),
            c_b_hat_prime.clone(),
            c_s.clone(),
        ])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        // DONE ABSORBING THE FIRST COMMITMENTS

        // generates alpha and x
        let mut challenge_for_p_round_1 = <Self as VerifierR1CSLite2<P, F, C>>::v_next_msg::<R>(
            1,
            &verifier_PP,
            &mut php_v_view,
            fs_rng,
        );

        let alpha = challenge_for_p_round_1.get_scalar("alpha").unwrap();
        let x = challenge_for_p_round_1.get_scalar("x").unwrap();
        //--------------------- END ROUND 1 ---------------------//

        //--------------------- ROUND 2 ---------------------//

        // recover the commits of q and r
        let c_q = prover_commits.get_prv_oracle("q").unwrap();
        let c_r = prover_commits.get_prv_oracle("r").unwrap();

        // ABSORBING THE SECOND COMMITMENTS
        let vec_comm_to_byte = C::extract_vec_commit_swh(vec![c_q.clone(), c_r.clone()])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        // DONE ABSORBING THE SECOND COMMITMENTS

        // Simulate the verifier to get the challenges
        // generates y
        let mut challenge_for_p_round_2 = <Self as VerifierR1CSLite2<P, F, C>>::v_next_msg::<R>(
            2,
            &verifier_PP,
            &mut php_v_view,
            fs_rng,
        );
        let y = challenge_for_p_round_2.get_scalar("y").unwrap();

        //--------------------- END ROUND 2 ---------------------//

        // extracting b_hat_prime_at_y and sigma from prover view
        // let b_hat_prime_at_y = prover_view.get_scalar("b_hat_prime_at_y").unwrap();
        // let r_prime_at_y = prover_view.get_scalar("r_prime_at_y").unwrap();
        let sigma = p_view.get_scalar("sigma").unwrap();

        //--------------------- ROUND 3 ---------------------//
        let c_q_prime = prover_commits.get_prv_oracle("q_prime_poly").unwrap();
        let c_r_prime = prover_commits.get_prv_oracle("r_prime_poly").unwrap();

        // ABSORBING THE THIRD COMMITMENTS
        // Add sigma to the FS RNG
        let vec_comm_to_byte =
            C::extract_vec_commit_swh(vec![c_q_prime.clone(), c_r_prime.clone()])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        fs_rng.absorb(&to_bytes![vec![sigma]].unwrap());
        // DONE ABSORBING THE THIRD COMMITMENTS
        //--------------------- END ROUND 3 ---------------------//

        Ok(php_v_view)
    }

    fn verifier_precompute<R: rand_core::RngCore + FiatShamirRng>(
        verifier_PP: &VerifierPP<P::Fr>,
        v_view: &mut View<P::Fr>,
        p_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_commits: &CView<P, C>,
        fs_rng: &mut R,
    ) -> (SPolynomial<P::Fr, SparseTerm>, C::CommT, P::Fr) {
        // Extract commits
        let c_a_hat_prime = prover_commits.get_prv_oracle("a_hat_prime").unwrap();
        let c_b_hat_prime = prover_commits.get_prv_oracle("b_hat_prime").unwrap();
        let c_s = prover_commits.get_prv_oracle("s").unwrap();
        let c_q = prover_commits.get_prv_oracle("q").unwrap();
        let c_r = prover_commits.get_prv_oracle("r").unwrap();

        // Extract scalars
        let y = v_view.get_scalar("y").unwrap();
        let x = v_view.get_scalar("x").unwrap();
        let alpha = v_view.get_scalar("alpha").unwrap();
        let b_hat_prime_at_y = p_view.get_scalar("b_hat_prime_at_y").unwrap();
        let sigma = p_view.get_scalar("sigma").unwrap();

        // Generate the values from the verifier PP
        let (n, m, l) = verifier_PP.cs.get_params();
        // regenerate the domains. Can this be avoided?
        let domain_h = GeneralEvaluationDomain::<P::Fr>::new(n).unwrap();
        let domain_k = GeneralEvaluationDomain::<P::Fr>::new(m).unwrap();
        let domain_k_size = domain_k.size();
        let LL = &domain_h.elements().collect::<Vec<P::Fr>>()[0..l];
        let z_L = vanishing_on_set(LL);
        let x_vec = verifier_PP.x_vec.clone();
        let mut x_prime = Vec::<P::Fr>::new();
        x_prime.push(P::Fr::one());
        x_prime.extend(x_vec.clone());

        let z_h_at_y = domain_h.evaluate_vanishing_polynomial(y);
        let z_h_at_x = domain_h.evaluate_vanishing_polynomial(x);
        let z_L_at_y = z_L.evaluate(&y);
        let lagrange_bivariate_at_xy = lambda_x(x, domain_h).evaluate(&y);
        let sum_x_vec_L = LL
            .iter()
            .fold(poly_const!(P::Fr, P::Fr::zero()), |acc, &tau| {
                acc + &poly_const!(P::Fr, x_prime[phi(&tau, domain_h).unwrap()])
                    * &lagrange_at_tau(domain_h, tau)
            });

        let sum_x_vec_L_at_y = sum_x_vec_L.evaluate(&y);

        // Same as the prover
        let constant_term = b_hat_prime_at_y
            * z_L_at_y
            * (sigma * sum_x_vec_L_at_y + alpha * lagrange_bivariate_at_xy)
            + alpha * lagrange_bivariate_at_xy
            + sum_x_vec_L_at_y * (sigma + lagrange_bivariate_at_xy);

        let c_constant = <C as CommitmentScheme<P>>::constant_commit(cs_pp, constant_term);

        let c_p =
            c_s + <C as CommitmentScheme<P>>::scale_comm(
                c_a_hat_prime.clone(),
                z_L_at_y * (lagrange_bivariate_at_xy + sigma),
            ) + <C as CommitmentScheme<P>>::scale_comm(
                c_a_hat_prime.clone(),
                b_hat_prime_at_y * z_L_at_y * z_L_at_y * sigma,
            ) + <C as CommitmentScheme<P>>::scale_comm(c_q.clone(), -z_h_at_y)
                + <C as CommitmentScheme<P>>::scale_comm(c_r.clone(), -y)
                + c_constant;

        // ABSORBING the scalars y, b_hat_at_y and 0
        fs_rng.absorb(&to_bytes![vec![&y, &b_hat_prime_at_y, &P::Fr::zero()]].unwrap());
        // DONE ABSORBING the scalars y, b_hat_at_y and 0

        //  FS RNG FOR FIRST EQ
        let mut rhoi = Vec::new();
        for i in 0..2 {
            rhoi.push(P::Fr::rand(fs_rng));
            fs_rng.absorb(&to_bytes![i as u32].unwrap());
        }

        let c_p_prime = <C as CommitmentScheme<P>>::scale_comm(c_p, rhoi[0])
            + <C as CommitmentScheme<P>>::scale_comm(c_b_hat_prime, rhoi[1]);

        let p_prime_at_y = b_hat_prime_at_y * rhoi[1];

        // Generates the values x^i y^j
        let mut xi_yj: Vec<Vec<P::Fr>> = Vec::new();
        for i in 0..3 {
            let mut x_yj: Vec<P::Fr> = Vec::new();
            for j in 0..3 {
                x_yj.push(x.pow([i]) * y.pow([j]));
            }
            xi_yj.push(x_yj);
        }

        // Creating the G polynomial and giving the commitments to the verifier
        // Coefficients for the MV poly
        let n2 = P::Fr::from((n * n) as u32);
        let sigma_over_K_n2: P::Fr = (sigma / P::Fr::from(domain_k_size as u32)) * n2;
        let z_H_x_z_H_y = z_h_at_y * z_h_at_x;

        // Move to the verifier side
        // Attribution of the number for the MV poly:
        // 0 => r'(X)
        // 1 => q'(X)
        // 2..=10 => cr_i,j(X)
        // 11..=19 => cr'_i,j(X)
        // 20..=23 => vcl_i,j(X)
        // 24..=27 => vcr_i,j(X)
        // 28 => z_K(X)
        let terms: Vec<(P::Fr, SparseTerm)> = vec![
            // cr_i,j(X)
            // TODO CHECK THE VALUES IN THE VECTOR FOR xy ARE IN THE CORRECT ORDER
            (sigma_over_K_n2 * xi_yj[0][0], SparseTerm::new(vec![(2, 1)])),
            (sigma_over_K_n2 * xi_yj[0][1], SparseTerm::new(vec![(3, 1)])),
            (sigma_over_K_n2 * xi_yj[0][2], SparseTerm::new(vec![(4, 1)])),
            (sigma_over_K_n2 * xi_yj[1][0], SparseTerm::new(vec![(5, 1)])),
            (sigma_over_K_n2 * xi_yj[1][1], SparseTerm::new(vec![(6, 1)])),
            (sigma_over_K_n2 * xi_yj[1][2], SparseTerm::new(vec![(7, 1)])),
            (sigma_over_K_n2 * xi_yj[2][0], SparseTerm::new(vec![(8, 1)])),
            (sigma_over_K_n2 * xi_yj[2][1], SparseTerm::new(vec![(9, 1)])),
            (
                sigma_over_K_n2 * xi_yj[2][2],
                SparseTerm::new(vec![(10, 1)]),
            ),
            // r'(X) * cr'_i,j(X)
            (n2 * xi_yj[0][0], SparseTerm::new(vec![(0, 1), (11, 1)])),
            (n2 * xi_yj[0][1], SparseTerm::new(vec![(0, 1), (12, 1)])),
            (n2 * xi_yj[0][2], SparseTerm::new(vec![(0, 1), (13, 1)])),
            (n2 * xi_yj[1][0], SparseTerm::new(vec![(0, 1), (14, 1)])),
            (n2 * xi_yj[1][1], SparseTerm::new(vec![(0, 1), (15, 1)])),
            (n2 * xi_yj[1][2], SparseTerm::new(vec![(0, 1), (16, 1)])),
            (n2 * xi_yj[2][0], SparseTerm::new(vec![(0, 1), (17, 1)])),
            (n2 * xi_yj[2][1], SparseTerm::new(vec![(0, 1), (18, 1)])),
            (n2 * xi_yj[2][2], SparseTerm::new(vec![(0, 1), (19, 1)])),
            // vcl_i,j(X)
            (-z_H_x_z_H_y * xi_yj[0][0], SparseTerm::new(vec![(20, 1)])),
            (-z_H_x_z_H_y * xi_yj[0][1], SparseTerm::new(vec![(21, 1)])),
            (-z_H_x_z_H_y * xi_yj[1][0], SparseTerm::new(vec![(22, 1)])),
            (-z_H_x_z_H_y * xi_yj[1][1], SparseTerm::new(vec![(23, 1)])),
            // vcr_i,j(X)
            (
                -z_H_x_z_H_y * xi_yj[0][0] * alpha,
                SparseTerm::new(vec![(24, 1)]),
            ),
            (
                -z_H_x_z_H_y * xi_yj[0][1] * alpha,
                SparseTerm::new(vec![(25, 1)]),
            ),
            (
                -z_H_x_z_H_y * xi_yj[1][0] * alpha,
                SparseTerm::new(vec![(26, 1)]),
            ),
            (
                -z_H_x_z_H_y * xi_yj[1][1] * alpha,
                SparseTerm::new(vec![(27, 1)]),
            ),
            // q'(X) * z_K(X)
            (-P::Fr::one(), SparseTerm::new(vec![(1, 1), (28, 1)])),
        ];

        // generate the MV poly
        // number of variable for the MV poly
        let num_var = 29;
        let G: SPolynomial<P::Fr, SparseTerm> = SPolynomial::from_coefficients_vec(num_var, terms);

        (G, c_p_prime, p_prime_at_y)
    }

    fn verifier_checks(
        verifier_PP: &VerifierPP<P::Fr>,
        cs_pp: &C::CS_PP,
        v_view: &mut View<P::Fr>,
        prover_commits: &CView<P, C>,
        vec_REL_commit: Vec<C::CommT>,
        vk_d: Vec<P::G2Projective>,
        prover_proofs: Vec<P::G1Projective>,
        G: SPolynomial<P::Fr, SparseTerm>,
        c_p_prime: C::CommT,
        p_prime_at_y: P::Fr,
    ) -> Result<(), CPError> {
        let y = v_view.get_scalar("y").unwrap();

        let c_q_prime = prover_commits.get_prv_oracle("q_prime_poly").unwrap();
        let c_r_prime = prover_commits.get_prv_oracle("r_prime_poly").unwrap();

        let proof_eq1 = prover_proofs[0];

        let check1 =
            <C as CPEval<P, C>>::verify(&cs_pp, &proof_eq1, c_p_prime, (&y, &p_prime_at_y))
                .unwrap();

        let num_var = G.num_vars();

        // Create a vector of G1 commits of length num_var_G filled with zeros
        //expect for the first to position that are the commits to r_prime and q_prime
        let mut comm_pi_g1 = vec![<C as CommitmentScheme<P>>::CommSwh::zero(); num_var];
        comm_pi_g1[0] = <C as CommitmentScheme<P>>::extract_commit_swh(c_r_prime.clone())?;
        comm_pi_g1[1] = <C as CommitmentScheme<P>>::extract_commit_swh(c_q_prime)?;

        // A vector of G2 commits with two zeros at the beginning and the rest are RE commits
        let mut comm_pi_g2 = Vec::new();
        comm_pi_g2.push(<C as CommitmentScheme<P>>::CommRel::zero());
        comm_pi_g2.push(<C as CommitmentScheme<P>>::CommRel::zero());
        comm_pi_g2.append(&mut <C as CommitmentScheme<P>>::extract_vec_commit_rel(
            vec_REL_commit.clone(),
        )?);

        // NECESSARY? if so compute the complementary commits in the prover side
        let _result_G_eq_check =
            <C as CPqeq<P, C>>::verify_Ghat_eval(&cs_pp, G, &comm_pi_g1, &comm_pi_g2).unwrap();

        let c_r = prover_commits.get_prv_oracle("r").unwrap();
        let c_r_star = prover_commits.get_prv_oracle("c_r_star").unwrap();
        let c_r_prime_star = prover_commits.get_prv_oracle("c_r_prime_star").unwrap();

        // Check individually each degree bound using CP2deg
        // Should the check be done using CP*deg ?
        let _result_deg_check_r =
            <C as CPdeg<P, C>>::verify_one_deg_bound(cs_pp, vk_d[0], c_r_star, c_r).unwrap();

        let _result_deg_check_r_prime =
            <C as CPdeg<P, C>>::verify_one_deg_bound(cs_pp, vk_d[1], c_r_prime_star, c_r_prime)
                .unwrap();

        Ok(())
    }

    // Only computes the values to be used in the batched checks in 7 pairings
    fn verifier_precompute_for_batched<R: rand_core::RngCore + FiatShamirRng>(
        verifier_PP: &VerifierPP<P::Fr>,
        v_view: &mut View<P::Fr>,
        p_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_commits: &CView<P, C>,
        fs_rng: &mut R,
    ) -> ((C::CommT, P::Fr), (Vec<P::Fr>, Vec<P::Fr>, Vec<P::Fr>)) {
        // Extract commits
        let c_a_hat_prime = prover_commits.get_prv_oracle("a_hat_prime").unwrap();
        let c_b_hat_prime = prover_commits.get_prv_oracle("b_hat_prime").unwrap();
        let c_s = prover_commits.get_prv_oracle("s").unwrap();
        let c_q = prover_commits.get_prv_oracle("q").unwrap();
        let c_r = prover_commits.get_prv_oracle("r").unwrap();

        // Extract scalars
        let y = v_view.get_scalar("y").unwrap();
        let x = v_view.get_scalar("x").unwrap();
        let alpha = v_view.get_scalar("alpha").unwrap();
        let b_hat_prime_at_y = p_view.get_scalar("b_hat_prime_at_y").unwrap();
        let sigma = p_view.get_scalar("sigma").unwrap();

        // Generate the values from the verifier PP
        let (n, m, l) = verifier_PP.cs.get_params();
        // regenerate the domains. Can this be avoided?
        let domain_h = GeneralEvaluationDomain::<P::Fr>::new(n).unwrap();
        let domain_k = GeneralEvaluationDomain::<P::Fr>::new(m).unwrap();
        let domain_k_size = domain_k.size();
        let LL = &domain_h.elements().collect::<Vec<P::Fr>>()[0..l];
        let z_L = vanishing_on_set(LL);
        let x_vec = verifier_PP.x_vec.clone();
        let mut x_prime = Vec::<P::Fr>::new();
        x_prime.push(P::Fr::one());
        x_prime.extend(x_vec.clone());

        let z_h_at_y = domain_h.evaluate_vanishing_polynomial(y);
        let z_h_at_x = domain_h.evaluate_vanishing_polynomial(x);
        let z_L_at_y = z_L.evaluate(&y);
        let lagrange_bivariate_at_xy = lambda_x(x, domain_h).evaluate(&y);
        let sum_x_vec_L = LL
            .iter()
            .fold(poly_const!(P::Fr, P::Fr::zero()), |acc, &tau| {
                acc + &poly_const!(P::Fr, x_prime[phi(&tau, domain_h).unwrap()])
                    * &lagrange_at_tau(domain_h, tau)
            });

        let sum_x_vec_L_at_y = sum_x_vec_L.evaluate(&y);

        // Same as the prover
        let constant_term = b_hat_prime_at_y
            * z_L_at_y
            * (sigma * sum_x_vec_L_at_y + alpha * lagrange_bivariate_at_xy)
            + alpha * lagrange_bivariate_at_xy
            + sum_x_vec_L_at_y * (sigma + lagrange_bivariate_at_xy);

        let c_constant = <C as CommitmentScheme<P>>::constant_commit(cs_pp, constant_term);

        let c_p =
            c_s + <C as CommitmentScheme<P>>::scale_comm(
                c_a_hat_prime.clone(),
                z_L_at_y * (lagrange_bivariate_at_xy + sigma),
            ) + <C as CommitmentScheme<P>>::scale_comm(
                c_a_hat_prime.clone(),
                b_hat_prime_at_y * z_L_at_y * z_L_at_y * sigma,
            ) + <C as CommitmentScheme<P>>::scale_comm(c_q.clone(), -z_h_at_y)
                + <C as CommitmentScheme<P>>::scale_comm(c_r.clone(), -y)
                + c_constant;

        // ABSORBING the scalars y, b_hat_at_y and 0
        fs_rng.absorb(&to_bytes![vec![&y, &b_hat_prime_at_y, &P::Fr::zero()]].unwrap());
        // DONE ABSORBING the scalars y, b_hat_at_y and 0

        //  FS RNG FOR FIRST EQ
        let mut rhoi = Vec::new();
        for i in 0..2 {
            rhoi.push(P::Fr::rand(fs_rng));
            fs_rng.absorb(&to_bytes![i as u32].unwrap());
        }

        let c_p_prime = <C as CommitmentScheme<P>>::scale_comm(c_p, rhoi[0])
            + <C as CommitmentScheme<P>>::scale_comm(c_b_hat_prime, rhoi[1]);

        let p_prime_at_y = b_hat_prime_at_y * rhoi[1];

        // Creating the G polynomial and giving the commitments to the verifier
        // Coefficients for the MV poly
        let n2 = P::Fr::from((n * n) as u32);
        let sigma_over_K_n2: P::Fr = (sigma / P::Fr::from(domain_k_size as u32)) * n2;
        let z_H_x_z_H_y = z_h_at_y * z_h_at_x;

        let mut coeff_cr = Vec::new();
        let mut coeff_cr_prime = Vec::new();
        let mut coeff_vcrL_vcrR = Vec::new();

        // Generates the values x^i y^j
        for i in 0..3 {
            let mut x_yj: Vec<P::Fr> = Vec::new();
            for j in 0..3 {
                x_yj.push(x.pow([i]) * y.pow([j]));
                coeff_cr.push(sigma_over_K_n2 * x.pow([i]) * y.pow([j]));
                coeff_cr_prime.push(n2 * x.pow([i]) * y.pow([j]));
                if i < 2 && j < 2 {
                    coeff_vcrL_vcrR.push(-z_H_x_z_H_y * x.pow([i]) * y.pow([j]));
                }
            }
        }

        (
            (c_p_prime, p_prime_at_y),
            (coeff_cr, coeff_cr_prime, coeff_vcrL_vcrR),
        )
    }

    // TO BE MODIFIED
    // To be added to the verifier of R1CSlite2
    // Full verifier checks in 7 pairings
    fn verifier_checks_batched(
        verifier_PP: &VerifierPP<P::Fr>,
        cs_pp: &C::CS_PP,
        v_view: &mut View<P::Fr>,
        prover_commits: &CView<P, C>,
        commit_RE_cr: Vec<C::CommT>,
        commit_RE_cr_p: Vec<C::CommT>,
        commit_RE_vcrL: Vec<C::CommT>,
        commit_RE_vcrR: Vec<C::CommT>,
        cr_coeff: Vec<P::Fr>,
        cr_prime_coeff: Vec<P::Fr>,
        vcrL_vcrR_coeff: Vec<P::Fr>,
        vk_d: Vec<P::G2Projective>,
        prover_proof: P::G1Projective,
        c_p_prime: C::CommT,
        p_prime_at_y: P::Fr,
    ) -> Result<(), CPError> {
        let (n, m, l) = verifier_PP.cs.get_params();
        let domain_k = GeneralEvaluationDomain::<P::Fr>::new(m).unwrap();
        let domain_k_size = domain_k.size();

        let y = v_view.get_scalar("y").unwrap();
        let alpha = v_view.get_scalar("alpha").unwrap();

        let c_q_prime = prover_commits.get_prv_oracle("q_prime_poly").unwrap();
        let c_r_prime = prover_commits.get_prv_oracle("r_prime_poly").unwrap();

        let proof_eq1 = prover_proof;

        let c_r = prover_commits.get_prv_oracle("r").unwrap();
        let c_r_star = prover_commits.get_prv_oracle("c_r_star").unwrap();
        let c_r_prime_star = prover_commits.get_prv_oracle("c_r_prime_star").unwrap();

        let c_star = vec![c_r_star.clone(), c_r_prime_star.clone()];
        let c_prime = vec![c_r.clone(), c_r_prime.clone()];

        let deg_bound_check_r = n - 2;
        let deg_bound_check_r_prime = domain_k_size - 2;

        let deg_bounds = vec![deg_bound_check_r, deg_bound_check_r_prime];

        let _result_deg_check_r_prime = <C as CPdeg<P, C>>::batch_verify_deg_bound_eq1(
            cs_pp,
            &vk_d,
            &deg_bounds,
            c_star,
            c_prime,
            proof_eq1,
            c_p_prime,
            (&y, &p_prime_at_y),
        )
        .unwrap();

        let _result_G_eq_check = <C as CPqeq<P, C>>::batch_verify_eq2(
            &cs_pp,
            c_r_prime,
            c_q_prime,
            domain_k_size,
            commit_RE_cr,
            cr_coeff,
            commit_RE_cr_p,
            cr_prime_coeff,
            commit_RE_vcrL,
            commit_RE_vcrR,
            vcrL_vcrR_coeff,
            alpha,
        )
        .unwrap();

        Ok(())
    }
}

impl<P, F, C> VerifierR1CSLite2x<P, F, C> for R1CSLite2x
where
    P: PairingEngine + PairingEngine<Fr = F>,
    F: FftField,
    C: CommitmentScheme<P, CS_PP = CS1_PP<P>> + CPEval<P, C>,
{
    fn v_next_msg<R: rand_core::RngCore + FiatShamirRng>(
        rnd_idx: usize,
        verifier_input: &VerifierPP<F>,
        v_view: &mut View<F>,
        rng: &mut R,
    ) -> View<F> {
        let mut p_view: View<F> = View::<F>::new();

        /* Common boilerplate code */
        let domain_h = GeneralEvaluationDomain::<F>::new(verifier_input.cs.get_params().0).unwrap();

        match rnd_idx {
            1 => {
                // Check the rng rand function
                let alpha = <F>::rand(rng);
                let x = <F>::rand(rng);

                add_to_views!(scalar, p_view, v_view, alpha);
                add_to_views!(scalar, p_view, v_view, x);
            }
            2 => {
                let y = domain_h.sample_element_outside_domain(rng);
                add_to_views!(scalar, p_view, v_view, y);
            }
            3 => {
                // do nothing
            }
            _ => {
                unimplemented!()
            }
        }

        p_view
    }

    fn verifier_rounds<R: rand_core::RngCore + FiatShamirRng>(
        verifier_PP: &VerifierPP<P::Fr>,
        p_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_commits: &CView<P, C>,
        fs_rng: &mut R,
    ) -> Result<View<P::Fr>, CPError> {
        // Useless view for now. Check if we need it at some point
        let mut php_v_view: View<P::Fr> = View::<P::Fr>::new();

        //--------------------- ROUND 1 ---------------------//

        // recover the commits of a_hat_prime and b_hat_prime and s
        let c_a_hat_prime = prover_commits.get_prv_oracle("a_hat_prime").unwrap();
        let c_b_hat_prime = prover_commits.get_prv_oracle("b_hat_prime").unwrap();
        let c_s = prover_commits.get_prv_oracle("s").unwrap();

        // ABSORBING THE FIRST COMMITMENTS
        let vec_comm_to_byte = C::extract_vec_commit_swh(vec![
            c_a_hat_prime.clone(),
            c_b_hat_prime.clone(),
            c_s.clone(),
        ])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        // DONE ABSORBING THE FIRST COMMITMENTS

        // generates alpha and x
        let mut challenge_for_p_round_1 = <Self as VerifierR1CSLite2x<P, F, C>>::v_next_msg::<R>(
            1,
            &verifier_PP,
            &mut php_v_view,
            fs_rng,
        );

        let alpha = challenge_for_p_round_1.get_scalar("alpha").unwrap();
        let x = challenge_for_p_round_1.get_scalar("x").unwrap();
        //--------------------- END ROUND 1 ---------------------//

        //--------------------- ROUND 2 ---------------------//

        // recover the commits of q and r
        let c_q = prover_commits.get_prv_oracle("q").unwrap();
        let c_r = prover_commits.get_prv_oracle("r").unwrap();

        // ABSORBING THE SECOND COMMITMENTS
        let vec_comm_to_byte = C::extract_vec_commit_swh(vec![c_q.clone(), c_r.clone()])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        // DONE ABSORBING THE SECOND COMMITMENTS

        // Simulate the verifier to get the challenges
        // generates y
        let mut challenge_for_p_round_2 = <Self as VerifierR1CSLite2x<P, F, C>>::v_next_msg::<R>(
            2,
            &verifier_PP,
            &mut php_v_view,
            fs_rng,
        );
        let y = challenge_for_p_round_2.get_scalar("y").unwrap();

        //--------------------- END ROUND 2 ---------------------//

        // extracting b_hat_prime_at_y and sigma from prover view
        let sigma = p_view.get_scalar("sigma").unwrap();

        //--------------------- ROUND 3 ---------------------//
        let c_q_prime = prover_commits.get_prv_oracle("q_prime_poly").unwrap();
        let c_r_prime = prover_commits.get_prv_oracle("r_prime_poly").unwrap();

        // ABSORBING THE THIRD COMMITMENTS
        // Add sigma to the FS RNG
        let vec_comm_to_byte =
            C::extract_vec_commit_swh(vec![c_q_prime.clone(), c_r_prime.clone()])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        fs_rng.absorb(&to_bytes![vec![sigma]].unwrap());
        // DONE ABSORBING THE THIRD COMMITMENTS
        //--------------------- END ROUND 3 ---------------------//

        Ok(php_v_view)
    }

    fn verifier_precompute<R: rand_core::RngCore + FiatShamirRng>(
        verifier_PP: &VerifierPP<P::Fr>,
        v_view: &mut View<P::Fr>,
        p_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        prover_commits: &CView<P, C>,
        fs_rng: &mut R,
    ) -> Result<((C::CommT, P::Fr), (C::CommT, P::Fr)), CPError> {
        // Extract commits
        let c_a_hat_prime = prover_commits.get_prv_oracle("a_hat_prime").unwrap();
        let c_b_hat_prime = prover_commits.get_prv_oracle("b_hat_prime").unwrap();
        let c_s = prover_commits.get_prv_oracle("s").unwrap();
        let c_q = prover_commits.get_prv_oracle("q").unwrap();
        let c_r = prover_commits.get_prv_oracle("r").unwrap();

        let commit_RE_cr = prover_commits.get_idx_oracle_matrix("cr").unwrap();
        let commit_RE_vcrR = prover_commits.get_idx_oracle_matrix("v_crR").unwrap();
        let commit_RE_vcrL = prover_commits.get_idx_oracle_matrix("v_crL").unwrap();

        // Extract commits
        let c_q_prime = prover_commits.get_prv_oracle("q_prime_poly").unwrap();
        let c_r_prime = prover_commits.get_prv_oracle("r_prime_poly").unwrap();
        let c_r_star = prover_commits.get_prv_oracle("c_r_star").unwrap();
        let c_r_prime_star = prover_commits.get_prv_oracle("c_r_prime_star").unwrap();

        // Extract scalars
        let y = v_view.get_scalar("y").unwrap();
        let x = v_view.get_scalar("x").unwrap();
        let alpha = v_view.get_scalar("alpha").unwrap();
        let sigma = p_view.get_scalar("sigma").unwrap();
        let b_hat_prime_at_y = p_view.get_scalar("b_hat_prime_at_y").unwrap();
        let r_prime_at_y2 = p_view.get_scalar("r_prime_at_y2").unwrap();

        // Generate the values from the verifier PP
        let (n, m, l) = verifier_PP.cs.get_params();
        // regenerate the domains. Can this be avoided?
        let domain_h = GeneralEvaluationDomain::<P::Fr>::new(n).unwrap();
        let domain_k = GeneralEvaluationDomain::<P::Fr>::new(m).unwrap();
        let domain_k_size = domain_k.size();
        let LL = &domain_h.elements().collect::<Vec<P::Fr>>()[0..l];
        let z_L = vanishing_on_set(LL);
        let x_vec = verifier_PP.x_vec.clone();
        let mut x_prime = Vec::<P::Fr>::new();
        x_prime.push(P::Fr::one());
        x_prime.extend(x_vec.clone());

        // VALUES FOR DEG CHECKS
        let deg_bound_check_r = cs_pp.deg_bound - n + 2;
        let deg_bound_check_r_prime = cs_pp.deg_bound - domain_k_size + 2;

        let z_h_at_y = domain_h.evaluate_vanishing_polynomial(y);
        let z_h_at_x = domain_h.evaluate_vanishing_polynomial(x);
        let z_L_at_y = z_L.evaluate(&y);
        let lagrange_bivariate_at_xy = lambda_x(x, domain_h).evaluate(&y);
        let sum_x_vec_L = LL
            .iter()
            .fold(poly_const!(P::Fr, P::Fr::zero()), |acc, &tau| {
                acc + &poly_const!(P::Fr, x_prime[phi(&tau, domain_h).unwrap()])
                    * &lagrange_at_tau(domain_h, tau)
            });

        let sum_x_vec_L_at_y = sum_x_vec_L.evaluate(&y);

        // Same as the prover
        let constant_term = b_hat_prime_at_y
            * z_L_at_y
            * (sigma * sum_x_vec_L_at_y + alpha * lagrange_bivariate_at_xy)
            + alpha * lagrange_bivariate_at_xy
            + sum_x_vec_L_at_y * (sigma + lagrange_bivariate_at_xy);

        let c_constant = <C as CommitmentScheme<P>>::constant_commit(cs_pp, constant_term);

        let c_p =
            c_s + <C as CommitmentScheme<P>>::scale_comm(
                c_a_hat_prime.clone(),
                z_L_at_y * (lagrange_bivariate_at_xy + sigma),
            ) + <C as CommitmentScheme<P>>::scale_comm(
                c_a_hat_prime.clone(),
                b_hat_prime_at_y * z_L_at_y * z_L_at_y * sigma,
            ) + <C as CommitmentScheme<P>>::scale_comm(c_q.clone(), -z_h_at_y)
                + <C as CommitmentScheme<P>>::scale_comm(c_r.clone(), -y)
                + c_constant;

        // ABSORBING the scalars y, b_hat_at_y and 0
        fs_rng.absorb(&to_bytes![vec![&y, &b_hat_prime_at_y, &P::Fr::zero()]].unwrap());
        // DONE ABSORBING the scalars y, b_hat_at_y and 0

        //  FS RNG FOR FIRST EQ
        let mut rhoi = Vec::new();
        for i in 0..2 {
            rhoi.push(P::Fr::rand(fs_rng));
            fs_rng.absorb(&to_bytes![i as u32].unwrap());
        }

        let c_p_prime_eq1 = <C as CommitmentScheme<P>>::scale_comm(c_p, rhoi[0])
            + <C as CommitmentScheme<P>>::scale_comm(c_b_hat_prime, rhoi[1]);

        let p_prime_eq1_at_y = b_hat_prime_at_y * rhoi[1];

        // Generates the values x^i y^j
        let mut xi_yj: Vec<Vec<P::Fr>> = Vec::new();
        for i in 0..3 {
            let mut x_yj: Vec<P::Fr> = Vec::new();
            for j in 0..3 {
                x_yj.push(x.pow([i]) * y.pow([j]));
            }
            xi_yj.push(x_yj);
        }

        let it_8_lin = 1..9;
        let it_3_lin = 1..4;

        let c_xy_cr = it_8_lin.clone().fold(
            <C as CommitmentScheme<P>>::scale_comm(
                commit_RE_cr[(0) as usize][(0) as usize].clone(),
                x.pow([0]) * y.pow([0]),
            ),
            |acc, i| {
                acc + <C as CommitmentScheme<P>>::scale_comm(
                    commit_RE_cr[(i / 3) as usize][(i % 3) as usize].clone(),
                    x.pow([i / 3]) * y.pow([i % 3]),
                )
            },
        );

        let c_xy_vcrLR = it_3_lin.fold(
            <C as CommitmentScheme<P>>::scale_comm(
                commit_RE_vcrL[(0) as usize][(0) as usize].clone()
                    + <C as CommitmentScheme<P>>::scale_comm(
                        commit_RE_vcrR[(0) as usize][(0) as usize].clone(),
                        alpha,
                    ),
                x.pow([0]) * y.pow([0]),
            ),
            |acc, i| {
                acc + <C as CommitmentScheme<P>>::scale_comm(
                    commit_RE_vcrL[(i / 2) as usize][(i % 2) as usize].clone()
                        + <C as CommitmentScheme<P>>::scale_comm(
                            commit_RE_vcrR[(i / 2) as usize][(i % 2) as usize].clone(),
                            alpha,
                        ),
                    x.pow([i / 2]) * y.pow([i % 2]),
                )
            },
        );

        // PRECOMPUT EQ2 DONE

        // ABSORBING the scalars c_r_star, c_r_prime_star, deg_bounds?
        let vec_comm_to_byte =
            C::extract_vec_commit_swh(vec![c_r_star.clone(), c_r_prime_star.clone()])?;
        fs_rng.absorb(
            &to_bytes![
                vec_comm_to_byte,
                vec![
                    &(deg_bound_check_r as u64),
                    &(deg_bound_check_r_prime as u64)
                ]
            ]
            .unwrap(),
        );

        // GENERATING THE NEW POINT y2
        let y2 = domain_h.sample_element_outside_domain(fs_rng);
        add_to_views!(scalar, v_view, y2);

        let n2 = P::Fr::from((n * n) as u32);
        let sigma_over_K_mul_n2: P::Fr = (sigma / P::Fr::from(domain_k_size as u32)) * n2;
        let coeff_term_cr = n2 * y2 * r_prime_at_y2 + sigma_over_K_mul_n2;

        let scaled_c_xy_cr = <C as CommitmentScheme<P>>::scale_comm(c_xy_cr.clone(), coeff_term_cr);
        let scaled_c_xy_vcrLR = <C as CommitmentScheme<P>>::scale_comm(
            c_xy_vcrLR.clone(),
            -domain_h.evaluate_vanishing_polynomial(x) * domain_h.evaluate_vanishing_polynomial(y),
        );
        let scaled_c_q_prime = <C as CommitmentScheme<P>>::scale_comm(
            c_q_prime.clone(),
            -domain_k.evaluate_vanishing_polynomial(y2),
        );

        let converted_scaled_c_xy_cr =
            <C as CommitmentScheme<P>>::convert_from_rel_to_swh(scaled_c_xy_cr.clone());
        let converted_scaled_c_xy_vcrLR =
            <C as CommitmentScheme<P>>::convert_from_rel_to_swh(scaled_c_xy_vcrLR.clone());

        // Computation of the commit to p from previous commits
        let c_p = converted_scaled_c_xy_cr + converted_scaled_c_xy_vcrLR + scaled_c_q_prime;

        let c_p_r_star =
            <C as CommitmentScheme<P>>::scale_comm(c_r.clone(), y2.pow([deg_bound_check_r as u64]))
                + <C as CommitmentScheme<P>>::scale_comm(c_r_star.clone(), -P::Fr::one());
        let c_p_r_prime_star =
            <C as CommitmentScheme<P>>::scale_comm(
                c_r_prime.clone(),
                y2.pow([deg_bound_check_r_prime as u64]),
            ) + <C as CommitmentScheme<P>>::scale_comm(c_r_prime_star.clone(), -P::Fr::one());

        // ASSEMBLING THE BATCH BY SUMING THE EQUATIONS FOR EQ2, r', DEG_R and DEG_R_PRIME

        // ABSORBING the scalars y2, r_prime_at_y2 and 0 and the commitments c_p, c_p_r_star and c_p_r_prime_star
        let vec_comm_to_byte = C::extract_vec_commit_swh(vec![
            c_p.clone(),
            c_p_r_star.clone(),
            c_p_r_prime_star.clone(),
        ])?;
        fs_rng.absorb(
            &to_bytes![
                &y2,
                vec_comm_to_byte,
                &r_prime_at_y2,
                &P::Fr::zero(),
                &P::Fr::zero()
            ]
            .unwrap(),
        );

        //  FS RNG FOR SECOND EQ
        let mut rhoi = Vec::new();
        for i in 0..4 {
            rhoi.push(P::Fr::rand(fs_rng));
            fs_rng.absorb(&to_bytes![i as u32].unwrap());
        }

        // Corresponding commitment
        let c_p_prime_eq2_degs = <C as CommitmentScheme<P>>::scale_comm(c_p.clone(), rhoi[0])
            + <C as CommitmentScheme<P>>::scale_comm(c_r_prime.clone(), rhoi[1])
            + <C as CommitmentScheme<P>>::scale_comm(c_p_r_star.clone(), rhoi[2])
            + <C as CommitmentScheme<P>>::scale_comm(c_p_r_prime_star.clone(), rhoi[3]);

        let p_prime_eq2_degs_at_y2 = rhoi[1] * r_prime_at_y2;

        Ok(
            (
            (c_p_prime_eq1, p_prime_eq1_at_y),
            (c_p_prime_eq2_degs, p_prime_eq2_degs_at_y2),
            )
        )
    }

    fn verifier_checks(
        verifier_PP: &VerifierPP<P::Fr>,
        v_view: &mut View<P::Fr>,
        cs_pp: &C::CS_PP,
        polys_eval: (P::Fr, P::Fr),
        commits_polys: (C::CommT, C::CommT),
        prover_proofs: (P::G1Projective, P::G1Projective),
        _phantom_view: &CView<P, C>,
    ) -> Result<(), CPError> {
        let y = v_view.get_scalar("y").unwrap();
        let y2 = v_view.get_scalar("y2").unwrap();

        let (c_p_eq1, c_p_eq2_degs) = commits_polys.clone();

        let (proof_eq1, proof_eq2_degs) = prover_proofs.clone();

        let (p_eq1_at_y, p_eq2_degs_at_y2) = polys_eval.clone();

        let _check_all = <C as CPEval<P, C>>::batch_verify_with_rng(
            &cs_pp,
            prover_proofs,
            commits_polys,
            ((y, p_eq1_at_y), (y2, p_eq2_degs_at_y2)),
        )
        .unwrap();

        Ok(())
    }
}
