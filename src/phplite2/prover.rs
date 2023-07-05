use crate::matrixutils::*;
use crate::php::*;

use ark_poly::{
    domain::EvaluationDomain,
    evaluations::univariate::Evaluations,
    polynomial::Polynomial,
    univariate::{DenseOrSparsePolynomial, DensePolynomial as DPolynomial},
    GeneralEvaluationDomain, MVPolynomial, UVPolynomial,
};

use ark_ff::{to_bytes, FftField, Field, One, PrimeField, Zero};

use ark_bls12_381::{Bls12_381, Fr};
use ark_ec::{PairingEngine, ProjectiveCurve};

// Import Randomness
use rand::thread_rng;
//use rand::seq::SliceRandom;
use ark_marlin::rng::FiatShamirRng;
use ark_marlin::rng::SimpleHashFiatShamirRng;
use ark_std::rand::Rng;
use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};
use rand::prelude::IteratorRandom;
// use blake2::Blake2s;
// use rand_chacha::ChaChaRng;

use itertools::Itertools;

use std::marker::PhantomData;
use std::ops::Mul;

use crate::phplite2::{R1CSLite2, R1CSLite2x};

// use crate::matrixutils::MatrixEncoding;

use super::view::*;

use crate::phplite2::*;

use crate::comms::*;
use crate::evalpolyproofs::*;
use crate::matrixutils::*;

use crate::poly_const;
use crate::x;

/// Parameters specific to R1CSLite2
const b_a: usize = 1;
const b_b: usize = 2;
const b_q: usize = 1;
const b_r: usize = 1;
const b_s: usize = 1;

pub struct ProverPP<F: FftField> {
    pub cs: CSLiteParams<F>,
    pub x_vec: Vec<F>,
    pub a_prime_poly: DPolynomial<F>,
    pub b_prime_poly: DPolynomial<F>,
}

pub trait ProverR1CSLite2<P, F, C>
where
    P: PairingEngine + PairingEngine<Fr = F>,
    F: FftField,
    C: CommitmentScheme<P, CS_PP = CS2_PP<P>> + CPEval<P, C> + CPdeg<P, C>,
{
    fn prover_first_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    );

    fn prover_second_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    );

    fn prover_third_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    );

    fn prover_rounds<R: rand_core::RngCore + FiatShamirRng>(
        prover_PP: &ProverPP<F>,
        verifier_PP: &VerifierPP<F>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<F>,
        v_view: &mut View<F>,
        c_p_view: &mut CView<P, C>,
        fs_rng: &mut R,
    ) -> Result<(), CPError>;

    fn prover_precomput<R: rand_core::RngCore + FiatShamirRng>(
        prover_PP: &ProverPP<P::Fr>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<F>,
        v_view: &mut View<F>,
        c_p_view: &mut CView<P, C>,
        fs_rng: &mut R,
    );

    fn prover_proofs(
        prover_PP: &ProverPP<F>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<F>,
        c_p_view: &mut CView<P, C>,
    ) -> Result<P::G1Projective, CPError>;
}

pub trait ProverR1CSLite2x<P, F, C>
where
    P: PairingEngine + PairingEngine<Fr = F>,
    F: FftField,
    C: CommitmentScheme<P, CS_PP = CS1_PP<P>> + CPEval<P, C>,
{
    fn prover_first_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    );

    fn prover_second_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    );

    fn prover_third_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    );

    fn prover_rounds<R: rand_core::RngCore + FiatShamirRng>(
        prover_PP: &ProverPP<F>,
        verifier_PP: &VerifierPP<F>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<F>,
        v_view: &mut View<F>,
        c_p_view: &mut CView<P, C>,
        fs_rng: &mut R,
    ) -> Result<(), CPError>;

    fn prover_precomput<R: rand_core::RngCore + FiatShamirRng>(
        prover_PP: &ProverPP<P::Fr>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<F>,
        v_view: &mut View<F>,
        c_p_view: &mut CView<P, C>,
        fs_rng: &mut R,
    ) -> Result<(), CPError>;

    fn prover_proofs(
        prover_PP: &ProverPP<F>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<F>,
        _phantom_view: &mut CView<P, C>,
    ) -> Result<(P::G1Projective, P::G1Projective), CPError>;
}


impl<P, F, C> ProverR1CSLite2<P, F, C> for R1CSLite2
where
    P: PairingEngine + PairingEngine<Fr = F>,
    F: FftField,
    C: CommitmentScheme<P, CS_PP = CS2_PP<P>> + CPEval<P, C> + CPdeg<P, C> + CPqeq<P, C>,
{
    fn prover_first_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    ) {
        /* Common boilerplate code */
        let (n, m, l) = prover_input.cs.get_params();
        let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let subdomain_h_minus_l = &domain_h.elements().collect::<Vec<F>>()[l..];

        let q_s = DPolynomial::<F>::rand(b_s + b_q - 1, rng);
        let r_s = DPolynomial::<F>::rand(b_r + b_q - 1, rng);
        let s = q_s.mul_by_vanishing_poly(domain_h) + &r_s * &x!(F);

        // s(X) = q_s(X)*Z_H(X) + r_s(X) * X
        let (quotient, remainder) = s.divide_by_vanishing_poly(domain_h).unwrap();
        assert_eq!(quotient, q_s);
        assert_eq!(remainder, r_s.mul(&x!(F)));

        let a_hat_prime = masking_polynomial(
            &prover_input.a_prime_poly,
            b_a + b_q,
            &subdomain_h_minus_l,
            rng,
        );
        let b_hat_prime = masking_polynomial(
            &prover_input.b_prime_poly,
            b_b + b_q,
            &subdomain_h_minus_l,
            rng,
        );

        assert!(a_hat_prime.degree() <= n - l + b_a + b_q - 1);
        assert!(b_hat_prime.degree() <= n - l + b_b + b_q - 1);

        // add items to both views
        add_to_views!(oracle, p_view, a_hat_prime);
        add_to_views!(oracle, p_view, b_hat_prime);
        add_to_views!(oracle, p_view, s);
    }

    // updates the view for the prover
    fn prover_second_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    ) {
        /* Common boilerplate code */
        let (n, m, l) = prover_input.cs.get_params();
        let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let encodingL = &prover_input.cs.getL().encode_matrix();
        let encodingR = &prover_input.cs.getR().encode_matrix();
        let x_vec = &prover_input.x_vec;
        let LL = &domain_h.elements().collect::<Vec<F>>()[0..l];
        let z_L = vanishing_on_set(LL);

        let mut x_prime = Vec::<F>::new();
        x_prime.push(F::one());
        x_prime.extend(x_vec);

        let a_hat_prime = p_view.get_prv_oracle("a_hat_prime").unwrap();
        let b_hat_prime = p_view.get_prv_oracle("b_hat_prime").unwrap();
        let s = p_view.get_prv_oracle("s").unwrap();

        let x = p_view.get_scalar("x").unwrap();
        let alpha = p_view.get_scalar("alpha").unwrap();

        // compute a_hat = a_hat_prime*z_L+sum(x_prime[i]*lagrange_on_subdomain)
        // compute b_hat = b_hat_prime*z_L+1

        let a_hat = a_hat_prime.mul(&z_L)
            + LL.iter().fold(poly_const!(F, F::zero()), |acc, &tau| {
                acc + &poly_const!(F, x_prime[phi(&tau, domain_h).unwrap()])
                    * &lagrange_at_tau(domain_h, tau)
                //lagrange_on_subdomain_at_tau(LL, tau)
            });
        let b_hat = b_hat_prime.mul(&z_L) + poly_const!(F, F::one());

        assert!(a_hat.degree() <= n + b_a + b_q - 1);
        assert!(b_hat.degree() <= n + b_b + b_q - 1);
        let l = lambda_x(x, domain_h);

        // compute V_LR(x,X,alpha) = V_L(x,X)+alpha*V_R(x,X)
        // exportar v_LR para la tercera ronda
        let v_LR = v_M_Y(&encodingL, x) + &poly_const!(F, alpha) * &v_M_Y(&encodingR, x);

        // compute p = (a_hat + alpha * b_hat) * lambda(x,X) + a_hat * b_hat * V_LR(x,X,alpha)
        let p = (a_hat.clone() + poly_const!(F, alpha).mul(&b_hat)).mul(&lambda_x(x, domain_h))
            + a_hat.mul(&b_hat).mul(&v_LR);

        let sumcheck = domain_h
            .elements()
            .fold(F::zero(), |acc, eta| acc + p.evaluate(&eta));

        // check sum(p(x)) = 0 over HH:
        assert!(sumcheck.is_zero());

        // compute q,r as s+p = q*z + r*X;
        let sp = s + p;
        let qr = sp.divide_by_vanishing_poly(domain_h);
        let (q, rx) = qr.unwrap();
        assert_eq!(rx.coeffs[0], F::zero());
        let r = &rx / &x!(F);

        // add items to both views
        add_to_views!(oracle, p_view, q);
        add_to_views!(oracle, p_view, r);
    }

    // returns the next message for the verifier and updates the view for the prover
    fn prover_third_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    ) {
        /* Common boilerplate code */
        let (n, m, l) = prover_input.cs.get_params();
        let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let encodingL = &prover_input.cs.getL().encode_matrix();
        let encodingR = &prover_input.cs.getR().encode_matrix();
        let domain_k = GeneralEvaluationDomain::<F>::new(m).unwrap();
        let domain_k_size = domain_k.size();
        let z_K = domain_k.vanishing_polynomial();

        let x = p_view.get_scalar("x").unwrap();
        let alpha = p_view.get_scalar("alpha").unwrap();
        let y = p_view.get_scalar("y").unwrap();

        // calculo de sigma
        let v_LR = v_M_Y(&encodingL, x) + &poly_const!(F, alpha) * &v_M_Y(&encodingR, x);
        let sigma = v_LR.evaluate(&y);
        // calculo de p'(x)
        let p_prime_poly = p_prime(&encodingL, &encodingR, alpha, x, y);
        // sum(p'(eta)) = sigma
        let sumcheck = domain_k
            .elements()
            .fold(F::zero(), |acc, eta| acc + p_prime_poly.evaluate(&eta));
        assert_eq!(sumcheck, sigma);

        let r_prime_poly_shifted =
            &p_prime_poly - &poly_const!(F, (sigma / F::from(domain_k_size as u32)));
        assert_eq!(r_prime_poly_shifted.coeffs[0], F::zero());
        let r_prime_poly = &r_prime_poly_shifted / &x!(F);

        let n2_poly = poly_const!(F, (F::from((n * n) as u32)));
        // xycr = sum(x^i y^j cr_ij(X) for i, j = 0..2)
        let it_9 = (0..3).cartesian_product(0..3);
        let it_4 = (0..2).cartesian_product(0..2);

        let v_crL = p_view.get_idx_oracle_matrix("v_crL").unwrap();
        let v_crR = p_view.get_idx_oracle_matrix("v_crR").unwrap();
        let cr = p_view.get_idx_oracle_matrix("cr").unwrap();
        let cr_p = p_view.get_idx_oracle_matrix("cr_p").unwrap();

        let xy_cr_cr_p = it_9.fold(poly_const!(F, F::zero()), |acc, (i, j)| {
            acc + poly_const!(F, x.pow([i]) * y.pow([j])).mul(
                &(&(&poly_const!(F, sigma / F::from(domain_k_size as u32))
                    * &cr[i as usize][j as usize])
                    + &(&r_prime_poly * &cr_p[i as usize][j as usize])),
            )
        });

        let xy_vcr = it_4.fold(poly_const!(F, F::zero()), |acc, (i, j)| {
            acc + DPolynomial::<F>::from_coefficients_slice(&[x.pow([i]) * y.pow([j])]).mul(
                &(&v_crL[i as usize][j as usize]
                    + &poly_const!(F, alpha).mul(&v_crR[i as usize][j as usize])),
            )
        });

        let t_poly = &n2_poly.mul(&xy_cr_cr_p)
            - &poly_const!(
                F,
                domain_h.evaluate_vanishing_polynomial(x)
                    * domain_h.evaluate_vanishing_polynomial(y)
            )
            .mul(&xy_vcr);

        // computation of q'(x)
        let (q_prime_poly, remainder) = DenseOrSparsePolynomial::from(t_poly)
            .divide_with_q_and_r(&z_K.into())
            .unwrap();
        assert!(remainder.is_zero());

        add_to_views!(oracle, p_view, q_prime_poly);
        add_to_views!(oracle, p_view, r_prime_poly);

        // Adding sigma to the view to use in the computation of the poly for the proof
        add_to_views!(scalar, p_view, sigma);
    }

    fn prover_rounds<R: rand_core::RngCore + FiatShamirRng>(
        prover_PP: &ProverPP<F>,
        verifier_PP: &VerifierPP<F>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<F>,
        v_view: &mut View<F>,
        c_p_view: &mut CView<P, C>,
        fs_rng: &mut R,
    ) -> Result<(), CPError> {
        //--------------------- ROUND 1 ---------------------//
        // generates a_hat_prime, b_hat_prime and s
        <Self as ProverR1CSLite2<P, F, C>>::prover_first_round::<R>(prover_PP, p_view, fs_rng);

        // Commit to the first polynomials and add it to the views
        let comms_round_1 = commit_oracles_dict_swh::<P, C>(&cs_pp, &p_view.prv_oracles);
        c_p_view.append_prv_comm_oracles(&comms_round_1);

        //Extract the commitments for updating the FS RNG
        let c_a_hat_prime = c_p_view.get_prv_oracle("a_hat_prime").unwrap();
        let c_b_hat_prime = c_p_view.get_prv_oracle("b_hat_prime").unwrap();
        let c_s = c_p_view.get_prv_oracle("s").unwrap();

        // Find a better way to absorb the commits?
        // ABSORBING THE FIRST COMMITMENTS
        let vec_comm_to_byte = C::extract_vec_commit_swh(vec![
            c_a_hat_prime.clone(),
            c_b_hat_prime.clone(),
            c_s.clone(),
        ])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        // DONE ABSORBING THE FIRST COMMITMENTS

        // Simulate the verifier first round to get the challenges
        // generates the scalars alpha and x
        let challenge_for_p_round_1 =
            <Self as VerifierR1CSLite2<P, F, C>>::v_next_msg::<R>(1, verifier_PP, v_view, fs_rng);
        p_view.append_view(challenge_for_p_round_1);
        //--------------------- END ROUND 1 ---------------------//

        //--------------------- ROUND 2 ---------------------//
        // generates q and r
        <Self as ProverR1CSLite2<P, F, C>>::prover_second_round::<R>(prover_PP, p_view, fs_rng);

        // commit to the polys
        let comms_round_2 = commit_oracles_dict_swh::<P, C>(&cs_pp, &p_view.prv_oracles);
        c_p_view.append_prv_comm_oracles(&comms_round_2);

        // recover polys and commits
        let c_q = c_p_view.get_prv_oracle("q").unwrap();
        let c_r = c_p_view.get_prv_oracle("r").unwrap();

        // ABSORBING THE SECOND COMMITMENTS
        let vec_comm_to_byte = C::extract_vec_commit_swh(vec![c_q.clone(), c_r.clone()])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        // DONE ABSORBING THE SECOND COMMITMENTS

        // Simulate the verifier second round to get the challenges
        // generates scalar y
        let challenge_for_p_round_2 =
            <Self as VerifierR1CSLite2<P, F, C>>::v_next_msg::<R>(2, verifier_PP, v_view, fs_rng);

        p_view.append_view(challenge_for_p_round_2);

        //--------------------- END ROUND 2 ---------------------//

        //--------------------- ROUND 3 ---------------------//
        // generates q_prime, r_prime and sigma
        <Self as ProverR1CSLite2<P, F, C>>::prover_third_round::<R>(prover_PP, p_view, fs_rng);

        // Commits to the polys
        let comms_round_3 = commit_oracles_dict_swh::<P, C>(&cs_pp, &p_view.prv_oracles);
        c_p_view.append_prv_comm_oracles(&comms_round_3);

        // Recover polys, commits and sigma
        let sigma = p_view.get_scalar("sigma").unwrap();
        let c_q_prime = c_p_view.get_prv_oracle("q_prime_poly").unwrap();
        let c_r_prime = c_p_view.get_prv_oracle("r_prime_poly").unwrap();

        // Add the value of sigma to the verifier view
        add_to_views!(scalar, v_view, sigma);

        // ABSORBING THE THIRD COMMITMENTS
        // Add sigma to the FS RNG
        let vec_comm_to_byte =
            C::extract_vec_commit_swh(vec![c_q_prime.clone(), c_r_prime.clone()])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        fs_rng.absorb(&to_bytes![vec![sigma]].unwrap());
        // DONE ABSORBING THE THIRD COMMITMENTS
        //--------------------- END ROUND 3 ---------------------//
        Ok(())
    }

    // Computes the polynomials and commitments necessary for the proof
    // and adds them to the views
    fn prover_precomput<R: rand_core::RngCore + FiatShamirRng>(
        prover_PP: &ProverPP<P::Fr>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<P::Fr>,
        v_view: &mut View<P::Fr>,
        c_p_view: &mut CView<P, C>,
        fs_rng: &mut R,
    ) {
        // Generate the values from the prover PP
        let (n, m, l) = prover_PP.cs.get_params();
        // regenerate the domains. Can this be avoided?
        let domain_h = GeneralEvaluationDomain::<P::Fr>::new(n).unwrap();
        let LL = &domain_h.elements().collect::<Vec<P::Fr>>()[0..l];
        let domain_k = GeneralEvaluationDomain::<P::Fr>::new(m).unwrap();
        let domain_k_size = domain_k.size();
        let z_L = vanishing_on_set(LL);
        let x_vec = prover_PP.x_vec.clone();
        let mut x_prime = Vec::<P::Fr>::new();
        x_prime.push(P::Fr::one());
        x_prime.extend(x_vec.clone());

        let a_hat_prime = p_view.get_prv_oracle("a_hat_prime").unwrap();
        let b_hat_prime = p_view.get_prv_oracle("b_hat_prime").unwrap();
        let s = p_view.get_prv_oracle("s").unwrap();
        let q = p_view.get_prv_oracle("q").unwrap();
        let r = p_view.get_prv_oracle("r").unwrap();

        let alpha = p_view.get_scalar("alpha").unwrap();

        let x = p_view.get_scalar("x").unwrap();
        let y = p_view.get_scalar("y").unwrap();
        let sigma = p_view.get_scalar("sigma").unwrap();

        let c_a_hat_prime = c_p_view.get_prv_oracle("a_hat_prime").unwrap();
        let c_b_hat_prime = c_p_view.get_prv_oracle("b_hat_prime").unwrap();
        let c_s = c_p_view.get_prv_oracle("s").unwrap();

        let c_q = c_p_view.get_prv_oracle("q").unwrap();
        let c_r = c_p_view.get_prv_oracle("r").unwrap();

        // Compute for the one shot eval of b(X) and the polynomial formed by the eq check
        let b_hat_prime_at_y = b_hat_prime.evaluate(&y);

        // Add the value of b at y to the verifier view
        add_to_views!(scalar, p_view, v_view, b_hat_prime_at_y);

        // Computing useful values
        let z_h_at_y = domain_h.evaluate_vanishing_polynomial(y);
        let z_h_at_x = domain_h.evaluate_vanishing_polynomial(x);
        let z_L_at_y = z_L.evaluate(&y);
        let lagrange_bivariate_at_xy = lambda_x(x, domain_h).evaluate(&y);

        // can be retrieved from the prover round execution
        let sum_x_vec_L = LL
            .iter()
            .fold(poly_const!(P::Fr, P::Fr::zero()), |acc, &tau| {
                acc + &poly_const!(P::Fr, x_prime[phi(&tau, domain_h).unwrap()])
                    * &lagrange_at_tau(domain_h, tau)
            });

        let sum_x_vec_L_at_y = sum_x_vec_L.evaluate(&y);

        // Constant term for the first eq check as b is evaluate on y
        let constant_term = b_hat_prime_at_y
            * z_L_at_y
            * (sigma * sum_x_vec_L_at_y + alpha * lagrange_bivariate_at_xy)
            + alpha * lagrange_bivariate_at_xy
            + sum_x_vec_L_at_y * (sigma + lagrange_bivariate_at_xy);

        // Creation of the poly p to check p(y) = 0 including b(y)
        let p =
            s + a_hat_prime.mul(&poly_const!(
                P::Fr,
                z_L_at_y * (lagrange_bivariate_at_xy + sigma)
            )) + a_hat_prime.mul(&poly_const!(
                P::Fr,
                b_hat_prime_at_y * z_L_at_y * z_L_at_y * sigma
            )) + q.mul(-z_h_at_y)
                + r.mul(-y)
                + poly_const!(P::Fr, constant_term);

        // simulated "commit" to the constant polynomial
        let c_constant = <C as CommitmentScheme<P>>::constant_commit(cs_pp, constant_term);

        // Computation of the commit to p from previous commits
        let c_p = c_s.clone()
            + <C as CommitmentScheme<P>>::scale_comm(
                c_a_hat_prime.clone(),
                z_L_at_y * (lagrange_bivariate_at_xy + sigma),
            )
            + <C as CommitmentScheme<P>>::scale_comm(
                c_a_hat_prime.clone(),
                b_hat_prime_at_y * z_L_at_y * z_L_at_y * sigma,
            )
            + <C as CommitmentScheme<P>>::scale_comm(c_q.clone(), -z_h_at_y)
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

        // Final poly for eval of two poly equalities
        let p_prime_eq1 =
            p.mul(&poly_const!(P::Fr, rhoi[0])) + b_hat_prime.mul(&poly_const!(P::Fr, rhoi[1]));
        // Corresponding commitment
        let c_p_prime_eq1 = <C as CommitmentScheme<P>>::scale_comm(c_p.clone(), rhoi[0])
            + <C as CommitmentScheme<P>>::scale_comm(c_b_hat_prime.clone(), rhoi[1]);

        // Computing the value of p prime at y to do the proof
        // p(y) = 0 thus its just the value of b
        let p_prime_eq1_at_y = rhoi[1] * b_hat_prime_at_y;

        // Add the value of b at y to the verifier view
        add_to_views!(oracle, p_view, p_prime_eq1);
        add_to_views!(scalar, p_view, p_prime_eq1_at_y);
        add_to_views!(commit, c_p_view, c_p_prime_eq1);

        let r_prime = p_view.get_prv_oracle("r_prime_poly").unwrap();

        let c_r_prime = c_p_view.get_prv_oracle("r_prime_poly").unwrap();

        let y = p_view.get_scalar("y").unwrap();

        let deg_bound_check_r = cs_pp.deg_bound - n + 2;
        let deg_bound_check_r_prime = cs_pp.deg_bound - domain_k_size + 2;

        let r_star = shift_by_n::<P::Fr>(deg_bound_check_r, &r);
        let r_prime_star = shift_by_n::<P::Fr>(deg_bound_check_r_prime, &r_prime);

        let c_r_star = <C as CommitmentScheme<P>>::commit_swh(&cs_pp, &r_star);
        let c_r_prime_star = <C as CommitmentScheme<P>>::commit_swh(&cs_pp, &r_prime_star);

        let p_r_star = &r.mul(&poly_const!(P::Fr, y.pow([deg_bound_check_r as u64]))) - &r_star;
        let p_r_prime_star = &r_prime
            .mul(&poly_const!(P::Fr, y.pow([deg_bound_check_r_prime as u64])))
            - &r_prime_star;

        let c_p_r_star =
            <C as CommitmentScheme<P>>::scale_comm(c_r.clone(), y.pow([deg_bound_check_r as u64]))
                + <C as CommitmentScheme<P>>::scale_comm(c_r_star.clone(), -P::Fr::one());
        let c_p_r_prime_star =
            <C as CommitmentScheme<P>>::scale_comm(
                c_r_prime.clone(),
                y.pow([deg_bound_check_r_prime as u64]),
            ) + <C as CommitmentScheme<P>>::scale_comm(c_r_prime_star.clone(), -P::Fr::one());

        let p_r_star_at_y = P::Fr::from(0 as u32);
        let p_r_prime_star_at_y = P::Fr::from(0 as u32);

        add_to_views!(oracle, p_view, p_r_star);
        add_to_views!(oracle, p_view, p_r_prime_star);
        add_to_views!(commit, c_p_view, c_r_star);
        add_to_views!(commit, c_p_view, c_r_prime_star);
    }

    fn prover_proofs(
        prover_PP: &ProverPP<P::Fr>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<P::Fr>,
        c_p_view: &mut CView<P, C>,
    ) -> Result<P::G1Projective, CPError> {
        let (n, m, l) = prover_PP.cs.get_params();
        let domain_k = GeneralEvaluationDomain::<P::Fr>::new(m).unwrap();
        let domain_k_size = domain_k.size();

        let p_prime_eq1 = p_view.get_prv_oracle("p_prime_eq1").unwrap();
        let p_prime_eq1_at_y = p_view.get_scalar("p_prime_eq1_at_y").unwrap();

        let r = p_view.get_prv_oracle("r").unwrap();
        let r_prime = p_view.get_prv_oracle("r_prime_poly").unwrap();

        let c_r = c_p_view.get_prv_oracle("r").unwrap();
        let c_r_prime = c_p_view.get_prv_oracle("r_prime_poly").unwrap();

        let y = p_view.get_scalar("y").unwrap();

        let p_r_star_at_y = P::Fr::from(0 as u32);
        let p_r_prime_star_at_y = P::Fr::from(0 as u32);

        let deg_bound_check_r = n - 2;
        let deg_bound_check_r_prime = domain_k_size - 2;

        // include a proof that b(y) = beta...
        let proof_eq1 =
            <C as CPEval<P, C>>::prove_poly_eval(&cs_pp, &p_prime_eq1, (&y, &p_prime_eq1_at_y))
                .unwrap();

        // Prove individually each degree bound
        // Proof via commit to r_star(X) = X^(d-d_r) * r(X)
        let (c_r_star, _) = <C as CPdeg<P, C>>::prove_one_deg_bound(
            cs_pp,
            deg_bound_check_r,
            vec![r.clone()],
            vec![<C as CommitmentScheme<P>>::extract_commit_swh(c_r)?],
            vec![P::Fr::one()], //vec![rho_vec[0]] not necessary if no batch is done
        )
        .unwrap();

        // Proof via commit to r_star(X) = X^(d-d_r_prime) * r_prime(X)
        let (c_r_prime_star, _) = <C as CPdeg<P, C>>::prove_one_deg_bound(
            cs_pp,
            deg_bound_check_r_prime,
            vec![r_prime.clone()],
            vec![<C as CommitmentScheme<P>>::extract_commit_swh(c_r_prime)?],
            vec![P::Fr::one()], //vec![rho_vec[1]] not necessary if no batch is done
        )
        .unwrap();

        add_to_views!(commit, c_p_view, c_r_star);
        add_to_views!(commit, c_p_view, c_r_prime_star);

        Ok(proof_eq1)
    }
}

impl<P, F, C> ProverR1CSLite2x<P, F, C> for R1CSLite2x
where
    P: PairingEngine + PairingEngine<Fr = F>,
    F: FftField,
    C: CommitmentScheme<P, CS_PP = CS1_PP<P>> + CPEval<P, C>,
{
    fn prover_first_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    ) {
        /* Common boilerplate code */
        let (n, m, l) = prover_input.cs.get_params();
        let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let subdomain_h_minus_l = &domain_h.elements().collect::<Vec<F>>()[l..];

        let q_s = DPolynomial::<F>::rand(b_s + b_q - 1, rng);
        let r_s = DPolynomial::<F>::rand(b_r + b_q - 1, rng);
        let s = q_s.mul_by_vanishing_poly(domain_h) + &r_s * &x!(F);

        // s(X) = q_s(X)*Z_H(X) + r_s(X) * X
        let (quotient, remainder) = s.divide_by_vanishing_poly(domain_h).unwrap();
        assert_eq!(quotient, q_s);
        assert_eq!(remainder, r_s.mul(&x!(F)));

        let a_hat_prime = masking_polynomial(
            &prover_input.a_prime_poly,
            b_a + b_q,
            &subdomain_h_minus_l,
            rng,
        );
        let b_hat_prime = masking_polynomial(
            &prover_input.b_prime_poly,
            b_b + b_q,
            &subdomain_h_minus_l,
            rng,
        );

        assert!(a_hat_prime.degree() <= n - l + b_a + b_q - 1);
        assert!(b_hat_prime.degree() <= n - l + b_b + b_q - 1);

        // add items to both views
        add_to_views!(oracle, p_view, a_hat_prime);
        add_to_views!(oracle, p_view, b_hat_prime);
        add_to_views!(oracle, p_view, s);
    }

    // returns the next message for the verifier and updates the view for the prover
    fn prover_second_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    ) {
        /* Common boilerplate code */
        let (n, m, l) = prover_input.cs.get_params();
        let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let encodingL = &prover_input.cs.getL().encode_matrix();
        let encodingR = &prover_input.cs.getR().encode_matrix();
        let x_vec = &prover_input.x_vec;
        let LL = &domain_h.elements().collect::<Vec<F>>()[0..l];
        let z_L = vanishing_on_set(LL);

        let mut x_prime = Vec::<F>::new();
        x_prime.push(F::one());
        x_prime.extend(x_vec);

        let a_hat_prime = p_view.get_prv_oracle("a_hat_prime").unwrap();
        let b_hat_prime = p_view.get_prv_oracle("b_hat_prime").unwrap();
        let s = p_view.get_prv_oracle("s").unwrap();

        let x = p_view.get_scalar("x").unwrap();
        let alpha = p_view.get_scalar("alpha").unwrap();

        // compute a_hat = a_hat_prime*z_L+sum(x_prime[i]*lagrange_on_subdomain)
        // compute b_hat = b_hat_prime*z_L+1

        let a_hat = a_hat_prime.mul(&z_L)
            + LL.iter().fold(poly_const!(F, F::zero()), |acc, &tau| {
                acc + &poly_const!(F, x_prime[phi(&tau, domain_h).unwrap()])
                    * &lagrange_at_tau(domain_h, tau)
                //lagrange_on_subdomain_at_tau(LL, tau)
            });
        let b_hat = b_hat_prime.mul(&z_L) + poly_const!(F, F::one());

        assert!(a_hat.degree() <= n + b_a + b_q - 1);
        assert!(b_hat.degree() <= n + b_b + b_q - 1);
        let l = lambda_x(x, domain_h);

        // compute V_LR(x,X,alpha) = V_L(x,X)+alpha*V_R(x,X)
        // exportar v_LR para la tercera ronda
        let v_LR = v_M_Y(&encodingL, x) + &poly_const!(F, alpha) * &v_M_Y(&encodingR, x);

        // compute p = (a_hat + alpha * b_hat) * lambda(x,X) + a_hat * b_hat * V_LR(x,X,alpha)
        let p = (a_hat.clone() + poly_const!(F, alpha).mul(&b_hat)).mul(&lambda_x(x, domain_h))
            + a_hat.mul(&b_hat).mul(&v_LR);

        let sumcheck = domain_h
            .elements()
            .fold(F::zero(), |acc, eta| acc + p.evaluate(&eta));

        // check sum(p(x)) = 0 over HH:
        assert!(sumcheck.is_zero());

        // compute q,r as s+p = q*z + r*X;
        let sp = s + p;
        let qr = sp.divide_by_vanishing_poly(domain_h);
        let (q, rx) = qr.unwrap();
        assert_eq!(rx.coeffs[0], F::zero());
        let r = &rx / &x!(F);

        // add items to both views
        add_to_views!(oracle, p_view, q);
        add_to_views!(oracle, p_view, r);
    }

    // returns the next message for the verifier and updates the view for the prover
    fn prover_third_round<R: rand_core::RngCore + FiatShamirRng>(
        prover_input: &ProverPP<F>,
        p_view: &mut View<F>,
        rng: &mut R,
    ) {
        /* Common boilerplate code */
        let (n, m, l) = prover_input.cs.get_params();
        let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let encodingL = &prover_input.cs.getL().encode_matrix();
        let encodingR = &prover_input.cs.getR().encode_matrix();
        let domain_k = GeneralEvaluationDomain::<F>::new(m).unwrap();
        let domain_k_size = domain_k.size();
        let z_K = domain_k.vanishing_polynomial();

        let x = p_view.get_scalar("x").unwrap();
        let alpha = p_view.get_scalar("alpha").unwrap();
        let y = p_view.get_scalar("y").unwrap();

        // calculo de sigma
        let v_LR = v_M_Y(&encodingL, x) + &poly_const!(F, alpha) * &v_M_Y(&encodingR, x);
        let sigma = v_LR.evaluate(&y);
        // calculo de p'(x)
        let p_prime_poly = p_prime(&encodingL, &encodingR, alpha, x, y);
        // sum(p'(eta)) = sigma
        // Shouldn't it be domain K and not H for the sumcheck?
        let sumcheck = domain_k
            .elements()
            .fold(F::zero(), |acc, eta| acc + p_prime_poly.evaluate(&eta));
        assert_eq!(sumcheck, sigma);
        // calculo de r'(x)
        let r_prime_poly_shifted =
            &p_prime_poly - &poly_const!(F, (sigma / F::from(domain_k_size as u32)));
        assert_eq!(r_prime_poly_shifted.coeffs[0], F::zero());
        let r_prime_poly = &r_prime_poly_shifted / &x!(F);
        // calculo de t(x)

        let n2_poly = poly_const!(F, (F::from((n * n) as u32)));
        // xycr = sum(x^i y^j cr_ij(X) for i, j = 0..2)
        let it_9 = (0..3).cartesian_product(0..3);
        let it_4 = (0..2).cartesian_product(0..2);

        let v_crL = p_view.get_idx_oracle_matrix("v_crL").unwrap();
        let v_crR = p_view.get_idx_oracle_matrix("v_crR").unwrap();
        let cr = p_view.get_idx_oracle_matrix("cr").unwrap();

        let xy_cr = it_9.fold(poly_const!(F, F::zero()), |acc, (i, j)| {
            acc + poly_const!(F, x.pow([i]) * y.pow([j])).mul(&cr[i as usize][j as usize])
        });

        let xy_cr_Xr = xy_cr
            .mul(&(&r_prime_poly * &x!(F) + poly_const!(F, sigma / F::from(domain_k_size as u32))));

        let xy_vcr = it_4.fold(poly_const!(F, F::zero()), |acc, (i, j)| {
            acc + DPolynomial::<F>::from_coefficients_slice(&[x.pow([i]) * y.pow([j])]).mul(
                &(&v_crL[i as usize][j as usize]
                    + &poly_const!(F, alpha).mul(&v_crR[i as usize][j as usize])),
            )
        });

        let t_poly = &n2_poly.mul(&xy_cr_Xr)
            - &poly_const!(
                F,
                domain_h.evaluate_vanishing_polynomial(x)
                    * domain_h.evaluate_vanishing_polynomial(y)
            )
            .mul(&xy_vcr);

        // calculo de q'(x)
        let (q_prime_poly, remainder) = DenseOrSparsePolynomial::from(t_poly)
            .divide_with_q_and_r(&z_K.into())
            .unwrap();
        assert!(remainder.is_zero());

        add_to_views!(oracle, p_view, q_prime_poly);
        add_to_views!(oracle, p_view, r_prime_poly);

        // Adding sigma to the view to use in the computation of the poly for the proof
        add_to_views!(scalar, p_view, sigma);
    }

    fn prover_rounds<R: rand_core::RngCore + FiatShamirRng>(
        prover_PP: &ProverPP<P::Fr>,
        verifier_PP: &VerifierPP<P::Fr>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<P::Fr>,
        v_view: &mut View<P::Fr>,
        c_p_view: &mut CView<P, C>,
        fs_rng: &mut R,
    ) -> Result<(), CPError> {
        //--------------------- ROUND 1 ---------------------//
        // generates a_hat_prime, b_hat_prime and s
        <Self as ProverR1CSLite2x<P, F, C>>::prover_first_round(&prover_PP, p_view, fs_rng);

        // Commit to the first polynomials and add it to the views
        let comms_round_1 = commit_oracles_dict_swh::<P, C>(&cs_pp, &p_view.prv_oracles);
        c_p_view.append_prv_comm_oracles(&comms_round_1);

        //Extract the commitments for updating the FS RNG
        let c_a_hat_prime = c_p_view.get_prv_oracle("a_hat_prime").unwrap();
        let c_b_hat_prime = c_p_view.get_prv_oracle("b_hat_prime").unwrap();
        let c_s = c_p_view.get_prv_oracle("s").unwrap();

        // Find a better way to absorb the commits?
        // ABSORBING THE FIRST COMMITMENTS
        let vec_comm_to_byte = C::extract_vec_commit_swh(vec![
            c_a_hat_prime.clone(),
            c_b_hat_prime.clone(),
            c_s.clone(),
        ])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        // DONE ABSORBING THE FIRST COMMITMENTS

        // Simulate the verifier first round to get the challenges
        // generates the scalars alpha and x
        let challenge_for_p_round_1 =
            <Self as VerifierR1CSLite2x<P, F, C>>::v_next_msg::<R>(1, &verifier_PP, v_view, fs_rng);
        p_view.append_view(challenge_for_p_round_1);
        //--------------------- END ROUND 1 ---------------------//

        //--------------------- ROUND 2 ---------------------//
        // generates q and r
        <Self as ProverR1CSLite2x<P, F, C>>::prover_second_round(&prover_PP, p_view, fs_rng);

        // commit to the polys
        let comms_round_2 = commit_oracles_dict_swh::<P, C>(&cs_pp, &p_view.prv_oracles);
        c_p_view.append_prv_comm_oracles(&comms_round_2);

        // recover polys and commits
        let c_q = c_p_view.get_prv_oracle("q").unwrap();
        let c_r = c_p_view.get_prv_oracle("r").unwrap();

        // ABSORBING THE SECOND COMMITMENTS
        let vec_comm_to_byte = C::extract_vec_commit_swh(vec![c_q.clone(), c_r.clone()])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        // DONE ABSORBING THE SECOND COMMITMENTS

        // Simulate the verifier second round to get the challenges
        // generates scalar y
        let challenge_for_p_round_2 =
            <Self as VerifierR1CSLite2x<P, F, C>>::v_next_msg::<R>(2, &verifier_PP, v_view, fs_rng);

        p_view.append_view(challenge_for_p_round_2);

        //--------------------- END ROUND 2 ---------------------//

        //--------------------- ROUND 3 ---------------------//
        // generates q_prime, r_prime and sigma
        <Self as ProverR1CSLite2x<P, F, C>>::prover_third_round(&prover_PP, p_view, fs_rng);

        // Commits to the polys
        let comms_round_3 = commit_oracles_dict_swh::<P, C>(&cs_pp, &p_view.prv_oracles);
        c_p_view.append_prv_comm_oracles(&comms_round_3);

        // Recover polys, commits and sigma
        let sigma = p_view.get_scalar("sigma").unwrap();
        let c_q_prime = c_p_view.get_prv_oracle("q_prime_poly").unwrap();
        let c_r_prime = c_p_view.get_prv_oracle("r_prime_poly").unwrap();

        // Add the value of sigma to the verifier view
        add_to_views!(scalar, p_view, v_view, sigma);

        // ABSORBING THE THIRD COMMITMENTS
        // Add sigma to the FS RNG
        let vec_comm_to_byte =
            C::extract_vec_commit_swh(vec![c_q_prime.clone(), c_r_prime.clone()])?;
        fs_rng.absorb(&to_bytes![vec_comm_to_byte].unwrap());
        fs_rng.absorb(&to_bytes![vec![sigma]].unwrap());
        // DONE ABSORBING THE THIRD COMMITMENTS
        //--------------------- END ROUND 3 ---------------------//

		Ok(())
    }

    fn prover_precomput<R: rand_core::RngCore + FiatShamirRng>(
        prover_PP: &ProverPP<P::Fr>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<P::Fr>,
        v_view: &mut View<P::Fr>,
        c_p_view: &mut CView<P, C>,
        fs_rng: &mut R,
    ) -> Result<(), CPError>
	where
        P: PairingEngine,
        C: CommitmentScheme<P> + CPEval<P, C>,
        R: rand_core::RngCore + FiatShamirRng,
    {
        // Generate the values from the prover PP
        let (n, m, l) = prover_PP.cs.get_params();
        // regenerate the domains. Can this be avoided?
        let domain_h = GeneralEvaluationDomain::<P::Fr>::new(n).unwrap();
        let LL = &domain_h.elements().collect::<Vec<P::Fr>>()[0..l];
        let domain_k = GeneralEvaluationDomain::<P::Fr>::new(m).unwrap();
        let domain_k_size = domain_k.size();
        let z_L = vanishing_on_set(LL);
        let x_vec = prover_PP.x_vec.clone();
        let mut x_prime = Vec::<P::Fr>::new();
        x_prime.push(P::Fr::one());
        x_prime.extend(x_vec.clone());

        let a_hat_prime = p_view.get_prv_oracle("a_hat_prime").unwrap();
        let b_hat_prime = p_view.get_prv_oracle("b_hat_prime").unwrap();
        let s = p_view.get_prv_oracle("s").unwrap();
        let q = p_view.get_prv_oracle("q").unwrap();
        let r = p_view.get_prv_oracle("r").unwrap();

        let alpha = p_view.get_scalar("alpha").unwrap();
        let x = p_view.get_scalar("x").unwrap();
        let y = p_view.get_scalar("y").unwrap();
        let sigma = p_view.get_scalar("sigma").unwrap();

        let c_a_hat_prime = c_p_view.get_prv_oracle("a_hat_prime").unwrap();
        let c_b_hat_prime = c_p_view.get_prv_oracle("b_hat_prime").unwrap();
        let c_s = c_p_view.get_prv_oracle("s").unwrap();

        let c_q = c_p_view.get_prv_oracle("q").unwrap();
        let c_r = c_p_view.get_prv_oracle("r").unwrap();

        // Compute for the one shot eval of b(X) and the polynomial formed by the eq check
        let b_hat_prime_at_y = b_hat_prime.evaluate(&y);

        // Add the value of b at y to the verifier view
        add_to_views!(scalar, p_view, v_view, b_hat_prime_at_y);

        // Computing useful values
        let z_h_at_y = domain_h.evaluate_vanishing_polynomial(y);
        let z_h_at_x = domain_h.evaluate_vanishing_polynomial(x);
        let z_L_at_y = z_L.evaluate(&y);
        let lagrange_bivariate_at_xy = lambda_x(x, domain_h).evaluate(&y);

        // can be retrieved from the prover round execution
        let sum_x_vec_L = LL
            .iter()
            .fold(poly_const!(P::Fr, P::Fr::zero()), |acc, &tau| {
                acc + &poly_const!(P::Fr, x_prime[phi(&tau, domain_h).unwrap()])
                    * &lagrange_at_tau(domain_h, tau)
            });

        let sum_x_vec_L_at_y = sum_x_vec_L.evaluate(&y);

        // Constant term for the first eq check as b is evaluate on y
        let constant_term = b_hat_prime_at_y
            * z_L_at_y
            * (sigma * sum_x_vec_L_at_y + alpha * lagrange_bivariate_at_xy)
            + alpha * lagrange_bivariate_at_xy
            + sum_x_vec_L_at_y * (sigma + lagrange_bivariate_at_xy);

        // Creation of the poly p to check p(y) = 0 including b(y)
        let p =
            s + a_hat_prime.mul(&poly_const!(
                P::Fr,
                z_L_at_y * (lagrange_bivariate_at_xy + sigma)
            )) + a_hat_prime.mul(&poly_const!(
                P::Fr,
                b_hat_prime_at_y * z_L_at_y * z_L_at_y * sigma
            )) + q.mul(-z_h_at_y)
                + r.mul(-y)
                + poly_const!(P::Fr, constant_term);

        // simulated "commit" to the constant polynomial
        let c_constant = <C as CommitmentScheme<P>>::constant_commit(cs_pp, constant_term);

        // Computation of the commit to p from previous commits
        let c_p = c_s.clone()
            + <C as CommitmentScheme<P>>::scale_comm(
                c_a_hat_prime.clone(),
                z_L_at_y * (lagrange_bivariate_at_xy + sigma),
            )
            + <C as CommitmentScheme<P>>::scale_comm(
                c_a_hat_prime.clone(),
                b_hat_prime_at_y * z_L_at_y * z_L_at_y * sigma,
            )
            + <C as CommitmentScheme<P>>::scale_comm(c_q.clone(), -z_h_at_y)
            + <C as CommitmentScheme<P>>::scale_comm(c_r.clone(), -y)
            + c_constant; //<C as CommitmentScheme<P>>::convert_const_commT_swh(c_constant);

        // ABSORBING the scalars y, b_hat_at_y and 0
        fs_rng.absorb(&to_bytes![vec![&y, &b_hat_prime_at_y, &P::Fr::zero()]].unwrap());
        // DONE ABSORBING the scalars y, b_hat_at_y and 0

        //  FS RNG FOR FIRST EQ
        let mut rhoi = Vec::new();
        for i in 0..2 {
            rhoi.push(P::Fr::rand(fs_rng));
            fs_rng.absorb(&to_bytes![i as u32].unwrap());
        }

        // Final poly for eval of two poly equalities
        let p_prime_eq1 =
            p.mul(&poly_const!(P::Fr, rhoi[0])) + b_hat_prime.mul(&poly_const!(P::Fr, rhoi[1]));
        // Corresponding commitment
        let c_p_prime_eq1 = <C as CommitmentScheme<P>>::scale_comm(c_p.clone(), rhoi[0])
            + <C as CommitmentScheme<P>>::scale_comm(c_b_hat_prime.clone(), rhoi[1]);

        // Computing the value of p prime at y to do the proof
        // p(y) = 0 thus its just the value of b
        let p_prime_eq1_at_y = rhoi[1] * b_hat_prime_at_y;

        // Add the value of b at y to the verifier view
        add_to_views!(oracle, p_view, p_prime_eq1);
        add_to_views!(scalar, p_view, p_prime_eq1_at_y);
        add_to_views!(commit, c_p_view, c_p_prime_eq1);

        // PRECOMPUT EQ1 DONE

        // VALUES FOR EQ2 CHECK AND DEG CHECKS

        let q_prime = p_view.get_prv_oracle("q_prime_poly").unwrap();
        let r_prime = p_view.get_prv_oracle("r_prime_poly").unwrap();

        let c_r_prime = c_p_view.get_prv_oracle("r_prime_poly").unwrap();
        let c_q_prime = c_p_view.get_prv_oracle("q_prime_poly").unwrap();

        let commit_RE_cr = c_p_view.get_idx_oracle_matrix("cr").unwrap();
        let commit_RE_vcrR = c_p_view.get_idx_oracle_matrix("v_crR").unwrap();
        let commit_RE_vcrL = c_p_view.get_idx_oracle_matrix("v_crL").unwrap();

        // VALUES FOR DEG CHECKS
        let deg_bound_check_r = cs_pp.deg_bound - n + 2;
        let deg_bound_check_r_prime = cs_pp.deg_bound - domain_k_size + 2;

        let r_star = shift_by_n::<P::Fr>(deg_bound_check_r, &r);
        let r_prime_star = shift_by_n::<P::Fr>(deg_bound_check_r_prime, &r_prime);

        let c_r_star = <C as CommitmentScheme<P>>::commit_swh(&cs_pp, &r_star);
        let c_r_prime_star = <C as CommitmentScheme<P>>::commit_swh(&cs_pp, &r_prime_star);

        add_to_views!(commit, c_p_view, c_r_star);
        add_to_views!(commit, c_p_view, c_r_prime_star);

        // GENERATING VALUES FOR EQ2 CHECK
        // The goal is to linearize the term X * r'(X) * sum(cr(X))
        // No need to do it for X so only prove the evaluation of r'(y) = y_r_prime
        // Then prove the equation using the term y * y_r_prime * sum(cr(X))

        let cr = p_view.get_idx_oracle_matrix("cr").unwrap();
        let v_crR = p_view.get_idx_oracle_matrix("v_crR").unwrap();
        let v_crL = p_view.get_idx_oracle_matrix("v_crL").unwrap();

        let it_9 = (0..3).cartesian_product(0..3);
        let it_8_lin = 1..9;
        let it_4 = (0..2).cartesian_product(0..2);
        let it_3_lin = 1..4;

        let xy_cr = it_9
            .clone()
            .fold(poly_const!(P::Fr, P::Fr::zero()), |acc, (i, j)| {
                acc + poly_const!(P::Fr, x.pow([i]) * y.pow([j])).mul(&cr[i as usize][j as usize])
            });

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

        let xy_vcrLR = it_4
            .clone()
            .fold(poly_const!(P::Fr, P::Fr::zero()), |acc, (i, j)| {
                acc + DPolynomial::<P::Fr>::from_coefficients_slice(&[x.pow([i]) * y.pow([j])]).mul(
                    &(&v_crL[i as usize][j as usize]
                        + &poly_const!(P::Fr, alpha).mul(&v_crR[i as usize][j as usize])),
                )
            });

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

        // GENERATE NEW EVALUATION POINT
        let y2 = domain_h.sample_element_outside_domain(fs_rng);
        add_to_views!(scalar, p_view, y2);

        // Compute for the one shot eval of b(X) and the polynomial formed by the eq check
        let r_prime_at_y2 = r_prime.evaluate(&y2);
        add_to_views!(scalar, p_view, v_view, r_prime_at_y2);

        let coeff_term_cr = P::Fr::from((n * n) as u32)
            * (y2 * r_prime_at_y2 + sigma / P::Fr::from(domain_k_size as u32));
        let coeff_term_cr_poly = poly_const!(P::Fr, coeff_term_cr); //n2_poly.mul(&poly_const!(P::Fr, y * r_prime_at_y +  sigma/P::Fr::from(domain_k_size as u32)));

        // Creation of the poly p to check p(y) = 0 including
        let p = coeff_term_cr_poly.mul(&xy_cr)
            + poly_const!(
                P::Fr,
                -domain_h.evaluate_vanishing_polynomial(x)
                    * domain_h.evaluate_vanishing_polynomial(y)
            )
            .mul(&xy_vcrLR)
            + q_prime.mul(&poly_const!(
                P::Fr,
                -domain_k.evaluate_vanishing_polynomial(y2)
            ));

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

        // DEG CHECKS POLYNOMIALS GENERATION
        let p_r_star = &r.mul(&poly_const!(P::Fr, y2.pow([deg_bound_check_r as u64]))) - &r_star;
        let p_r_prime_star = &r_prime.mul(&poly_const!(
            P::Fr,
            y2.pow([deg_bound_check_r_prime as u64])
        )) - &r_prime_star;

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

        // Final poly for eval of two poly equalities
        let p_prime_eq2_degs = p.mul(&poly_const!(P::Fr, rhoi[0]))
            + r_prime.mul(&poly_const!(P::Fr, rhoi[1]))
            + p_r_star.mul(&poly_const!(P::Fr, rhoi[2]))
            + p_r_prime_star.mul(&poly_const!(P::Fr, rhoi[3]));

        // Corresponding commitment
        let c_p_prime_eq2_degs = <C as CommitmentScheme<P>>::scale_comm(c_p.clone(), rhoi[0])
            + <C as CommitmentScheme<P>>::scale_comm(c_r_prime.clone(), rhoi[1])
            + <C as CommitmentScheme<P>>::scale_comm(c_p_r_star.clone(), rhoi[2])
            + <C as CommitmentScheme<P>>::scale_comm(c_p_r_prime_star.clone(), rhoi[3]);

        let p_prime_eq2_degs_at_y2 = rhoi[1] * r_prime_at_y2;

        add_to_views!(oracle, p_view, p_prime_eq2_degs);
        add_to_views!(scalar, p_view, p_prime_eq2_degs_at_y2);
        add_to_views!(commit, c_p_view, c_p_prime_eq2_degs);

		Ok(())
    }

    fn prover_proofs(
        prover_PP: &ProverPP<P::Fr>,
        cs_pp: &C::CS_PP,
        p_view: &mut View<P::Fr>,
        _phantom_view: &mut CView<P, C>,
    ) -> Result<(P::G1Projective, P::G1Projective), CPError>
    where
        P: PairingEngine,
        C: CommitmentScheme<P, CS_PP = CS1_PP<P>> + CPEval<P, C>,
    {
        let p_prime_eq1 = p_view.get_prv_oracle("p_prime_eq1").unwrap();
        let p_prime_eq2_degs = p_view.get_prv_oracle("p_prime_eq2_degs").unwrap();
        let p_prime_eq1_at_y = p_view.get_scalar("p_prime_eq1_at_y").unwrap();
        let p_prime_eq2_degs_at_y2 = p_view.get_scalar("p_prime_eq2_degs_at_y2").unwrap();

        let y = p_view.get_scalar("y").unwrap();
        let y2 = p_view.get_scalar("y2").unwrap();

        // include a proof that b(y) = beta...
        let proof_eq1 =
            <C as CPEval<P, C>>::prove_poly_eval(&cs_pp, &p_prime_eq1, (&y, &p_prime_eq1_at_y))?;

        // include a proof that r'(y) = r_prime_at_y...
        let proof_eq2_degs = <C as CPEval<P, C>>::prove_poly_eval(
            &cs_pp,
            &p_prime_eq2_degs,
            (&y2, &p_prime_eq2_degs_at_y2),
        )?;

        Ok((proof_eq1, proof_eq2_degs))
    }
}
