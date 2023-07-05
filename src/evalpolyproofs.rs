use ark_ec::{prepare_g1, prepare_g2, AffineCurve, PairingEngine, ProjectiveCurve};

use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};

use ark_ff::{FftField, Field, One, PrimeField, Zero};

use ark_ec::msm::VariableBaseMSM;

use ark_poly::{
    domain::EvaluationDomain,
    evaluations::univariate::Evaluations,
    multivariate::{SparsePolynomial as SPolynomial, SparseTerm, Term},
    univariate::{DenseOrSparsePolynomial, DensePolynomial as DPolynomial},
    GeneralEvaluationDomain, MVPolynomial, UVPolynomial,
};

use crate::comms::*;
use crate::matrixutils::*;
use crate::poly_const;

use std::ops::Add;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;

use itertools::izip;
use std::iter::zip;

pub trait CPEval<P, C>: CommitmentScheme<P>
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    // Compute the value w(s) = (p(s) - p(a)) / (s - a)
    fn prove_poly_eval(
        ck: &Self::CS_PP,
        p: &DPolynomial<P::Fr>,
        x: (&P::Fr, &P::Fr), // (a, p(a))
    ) -> Result<P::G1Projective, CPError>;

    // verify with computation of the two pairing and check for their equality
    fn verify_with_eq_check(
        ck: &Self::CS_PP,
        prf: &P::G1Projective,
        comm: Self::CommT,
        a_b: (&P::Fr, &P::Fr),
    ) -> Result<(), CPError>;

    // optimized verify doing two pairings and check for equality to 1
    fn verify(
        ck: &Self::CS_PP,
        prf: &P::G1Projective,
        comm: Self::CommT,
        a_b: (&P::Fr, &P::Fr),
    ) -> Result<(), CPError>;

    // optimized verify doing two pairings with rng and check for equality to 1
    fn verify_with_rng(
        ck: &Self::CS_PP,
        prf: P::G1Projective,
        comm: Self::CommT,
        a_b: (&P::Fr, &P::Fr),
    ) -> Result<(), CPError>;

    // optimized verify doing two pairings with rng and
    // batching two proofs and check for equality to 1
    fn batch_verify_with_rng(
        ck: &Self::CS_PP,
        prfs: (P::G1Projective, P::G1Projective),
        comms: (Self::CommT, Self::CommT),
        a_b: ((P::Fr, P::Fr), (P::Fr, P::Fr)),
    ) -> Result<(), CPError>;
}

// Should correspond to CP_eval from the article for evaluation of several polynomials
pub trait CPEvals<P, C>: CommitmentScheme<P>
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    // Compute the value w(s) = (p(s) - p(a)) / (s - a)
    fn prove_poly_eval(
        ck: &Self::CS_PP,
        p: &DPolynomial<P::Fr>,
        x: (&P::Fr, &P::Fr),
    ) -> Result<P::G1Projective, CPError>;

    // Several calls to prove_poly_eval for different polys
    // but on the same point a
    fn multiple_evals_on_same_point(
        ck: &Self::CS_PP,
        a_bi: (P::Fr, Vec<P::Fr>),
        pi: Vec<DPolynomial<P::Fr>>,
        ci: Vec<Self::CommT>,
        rho: Vec<P::Fr>,
    ) -> Result<(P::G1Projective, P::Fr, Self::CommT), CPError>;

    // Several calls to multiple_evals_on_same_point for different polys
    // and the different points ai
    fn multiple_evals(
        ck: &Self::CS_PP,
        ai_bi: (&Vec<P::Fr>, Vec<Vec<P::Fr>>),
        pi: Vec<Vec<DPolynomial<P::Fr>>>,
        ci: Vec<Vec<Self::CommT>>,
        rhoi: Vec<Vec<P::Fr>>,
    ) -> Result<Vec<(P::G1Projective, P::Fr, Self::CommT)>, CPError>;
}

// Should correspond to CP_deg_star from the article for evaluation of several polynomials
pub trait CPdeg<P, C>: CommitmentScheme<P>
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    // Used by the RE select powers of g2
    // to give to the verifier
    fn derive(ck: &Self::CS_PP, deg_bounds: &Vec<usize>) -> Result<Vec<P::G2Projective>, CPError>;

    // verify function for the degree checks
    fn verify_one_deg_bound(
        ck: &Self::CS_PP,
        vk_d: P::G2Projective, // Not vec because checking one deg bound
        c_star: Self::CommT,
        c_prime: Self::CommT,
    ) -> Result<(), CPError>;

    // verify function for the degree checks
    // checks several degree bounds
    fn verify_deg_bound(
        ck: &Self::CS_PP,
        vk_d: &Vec<P::G2Projective>,
        deg_bounds: &Vec<usize>,
        c_star: Vec<Self::CommT>,
        c_prime: Vec<Self::CommT>,
    ) -> Result<(), CPError>;

    // Batch verification for CS2 eq1 and degree bounds
    // performs the check in 4 pairings using rng and batching
    fn batch_verify_deg_bound_eq1(
        ck: &Self::CS_PP,
        vk_d: &Vec<P::G2Projective>,
        deg_bounds: &Vec<usize>,
        c_star: Vec<Self::CommT>,
        c_prime: Vec<Self::CommT>,
        prf: P::G1Projective,
        comm_eq1: Self::CommT,
        a_b: (&P::Fr, &P::Fr),
    ) -> Result<(), CPError>;

    // Generates commitments to prove one deg bound
    // of several polynomials
    fn prove_one_deg_bound(
        ck: &Self::CS_PP,
        deg_bound: usize,
        polys: Vec<DPolynomial<P::Fr>>,
        commits: Vec<Self::CommSwh>,
        rho_vec: Vec<P::Fr>,
    ) -> Result<(Self::CommT, Self::CommT), CPError>;

    // Generates commitments to prove deg bounds
    // of several polynomials
    fn prove_deg_bound(
        ck: &Self::CS_PP,
        deg_bounds: &Vec<usize>,
        polys: Vec<Vec<DPolynomial<P::Fr>>>,
        commits: Vec<Vec<Self::CommT>>,
        rho_vec: Vec<Vec<P::Fr>>,
    ) -> Result<(Vec<Self::CommT>, Vec<Self::CommT>), CPError>;
}

// Should correspond to CP_eval from the paper for evaluation of several polynomials
pub trait CPqeq<P, C>: CommitmentScheme<P>
where
    P: PairingEngine,
    C: CommitmentScheme<P>,
{
    // Check (2) in the Verify QEQ
    // commits_g1 and commits_g2 contain the lists of commits for pairing check
    // Generic function not used in the implementation of
    // Lunar and LunarLite
    fn verify_complementary_commits(
        ck: &Self::CS_PP,
        commits_g1: &Vec<Self::CommSwh>,
        commits_g2: &Vec<Self::CommRel>,
    ) -> Result<(), CPError>;

    // Check (3) in the Verify QEQ
    // commits_g1 and commits_g2 contain the lists of commits to evaluate the Ghat poly
    // Generic function not used in the implementation of
    // Lunar and LunarLite
    fn verify_Ghat_eval(
        ck: &Self::CS_PP,
        g_hat: SPolynomial<P::Fr, SparseTerm>,
        commits_g1: &Vec<Self::CommSwh>,
        commits_g2: &Vec<Self::CommRel>,
    ) -> Result<(), CPError>;

    // compute the commits in the complementary group for sets of polys
    // Generic function not used in the implementation of
    // Lunar and LunarLite
    fn compute_complementary_commits(
        ck: &Self::CS_PP,
        p_to_g1: Vec<&DPolynomial<P::Fr>>, // poly to commit to g1
        p_to_g2: Vec<&DPolynomial<P::Fr>>, // poly to commit to g2
    ) -> (Vec<Self::CommSwh>, Vec<Self::CommRel>);

    // Function performing an optimized check of the eq2
    // using batching and rng in only 3 pairings
    fn batch_verify_eq2(
        ck: &Self::CS_PP,
        c_r_prime: Self::CommT,
        c_q_prime: Self::CommT,
        domain_k_size: usize,
        c_cr: Vec<Self::CommT>,
        cr_coeff: Vec<P::Fr>,
        c_cr_prime: Vec<Self::CommT>,
        cr_prime_coeff: Vec<P::Fr>,
        c_vcrL: Vec<Self::CommT>,
        c_vcrR: Vec<Self::CommT>,
        vcl_vcr_coeff: Vec<P::Fr>,
        alpha: P::Fr,
    ) -> Result<(), CPError>;
}

impl<P: PairingEngine, C: CommitmentScheme<P>> CPEval<P, C> for CS1 {
    fn prove_poly_eval(
        ck: &Self::CS_PP,
        p: &DPolynomial<P::Fr>,
        x: (&P::Fr, &P::Fr),
    ) -> Result<P::G1Projective, CPError> {
        let (i, phi_i) = x;
        let psi = kate_division::<P::Fr>(p, *i, *phi_i);

        let w_i = eval_on_pnt_in_grp::<P::G1Projective>(&psi, &ck.pws_g1);

        // No implementation of is_on_curve for P::G1Projective
        // in current arkworks version
        // if w_affine.is_on_curve() {
        //     Ok(w_i)
        // } else {
        //     Err(CPError::BadPrfInput)
        // }

        Ok(w_i)
    } // end prove

    // verify function doing two pairings and checking their equality
    fn verify_with_eq_check(
        ck: &Self::CS_PP,
        prf: &P::G1Projective,
        comm: Self::CommT,
        a_b: (&P::Fr, &P::Fr), // stands for (a, b) in the pairing describe in the paper
    ) -> Result<(), CPError> {
        let (a, b) = a_b;

        let commit = <Self as CommitmentScheme<P>>::extract_commit_swh(comm)?;

        // Compute the value of b in G1
        let mut b_in_G1 = ck.g1;
        b_in_G1.mul_assign(*b);

        // Compute the value of c - b in G1
        let lhs_pairing2 = commit - b_in_G1;

        // get the value of a in G2
        let mut a_in_G2 = ck.g2;
        a_in_G2.mul_assign(*a);

        // Compute the value s - a in G2
        let rhs_pairing1 = ck.alpha_g2 - a_in_G2;

        let term1 = P::pairing(*prf, rhs_pairing1);
        let term2 = P::pairing(lhs_pairing2, ck.g2);

        if term1.eq(&term2) {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }

    // optimized verify doing two pairings and check for equality to 1
    fn verify(
        ck: &Self::CS_PP,
        prf: &P::G1Projective,
        comm: Self::CommT,
        a_b: (&P::Fr, &P::Fr), // stands for (a, b) in the pairing describe in the paper
    ) -> Result<(), CPError> {
        let (a, b) = a_b;

        let commit = <Self as CommitmentScheme<P>>::extract_commit_swh(comm)?;

        // Compute the value of b in G1
        let mut b_in_G1 = ck.g1;
        b_in_G1.mul_assign(*b);

        // Compute the value of c - b in G1
        let lhs_pairing2 = b_in_G1 - commit;

        // get the value of a in G2
        let mut a_in_G2 = ck.g2;
        a_in_G2.mul_assign(*a);

        // Compute the value s - a in G2
        let rhs_pairing1 = ck.alpha_g2 - a_in_G2;

        let mut list_pairings: Vec<(P::G1Prepared, P::G2Prepared)> = Vec::new();
        list_pairings.push((
            P::G1Prepared::from((*prf).into()),
            P::G2Prepared::from(rhs_pairing1.into()),
        ));
        list_pairings.push((
            P::G1Prepared::from(lhs_pairing2.into()),
            P::G2Prepared::from(ck.g2.into()),
        ));

        let pairing_result = P::product_of_pairings(&list_pairings);
        let result = pairing_result.is_one();

        if result {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }

    // optimized verify doing two pairings with rng and check for equality to 1
    fn verify_with_rng(
        ck: &Self::CS_PP,
        prf: P::G1Projective,
        comm: Self::CommT,
        a_b: (&P::Fr, &P::Fr), // stands for (a, b) in the pairing describe in the paper
    ) -> Result<(), CPError> {
        let rng = &mut ark_std::test_rng();
        //  FS RNG FOR SECOND EQ
        let rho = P::Fr::rand(rng);

        let (a, b) = a_b;

        let comm_rho = <Self as CommitmentScheme<P>>::scale_comm(comm, rho);
        let commit_for_pairing = <Self as CommitmentScheme<P>>::extract_commit_swh(comm_rho)?;

        let mut proof_rho = prf;
        proof_rho.mul_assign(rho);

        // Compute the value of b in G1
        let mut b_in_G1 = ck.g1;
        b_in_G1.mul_assign(*b);
        b_in_G1.mul_assign(rho);

        // Compute the value of c - b in G1
        let lhs_pairing2 = commit_for_pairing - b_in_G1;

        // get the value of a in G2
        let mut a_in_G2 = ck.g2;
        a_in_G2.mul_assign(*a);

        // Compute the value s - a in G2
        let rhs_pairing1 = ck.alpha_g2 - a_in_G2;

        let mut list_pairings: Vec<(P::G1Prepared, P::G2Prepared)> = Vec::new();
        list_pairings.push((
            P::G1Prepared::from((-proof_rho).into()),
            P::G2Prepared::from(rhs_pairing1.into()),
        ));
        list_pairings.push((
            P::G1Prepared::from(lhs_pairing2.into()),
            P::G2Prepared::from(ck.g2.into()),
        ));

        let pairing_result = P::product_of_pairings(&list_pairings);

        let result = pairing_result.is_one();

        if result {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }

    // optimized verify doing two pairings with rng and batching proofs and check for equality to 1
    fn batch_verify_with_rng(
        ck: &Self::CS_PP,
        prfs: (P::G1Projective, P::G1Projective),
        comms: (Self::CommT, Self::CommT),
        a_b: ((P::Fr, P::Fr), (P::Fr, P::Fr)), // stands for (a, b) in the pairing describe in the paper
    ) -> Result<(), CPError> {
        // Verifier RNG to make sure the proof is correct
        let rng = &mut ark_std::test_rng();
        let rho1 = P::Fr::rand(rng);
        let rho2 = P::Fr::rand(rng);

        // UNPACKING
        // Here yi is the point and xi the evaluation
        let ((y1, x1), (y2, x2)) = a_b;
        let (prf1, prf2) = prfs;
        let (comm1, comm2) = comms;

        let comm1_rho1 = <Self as CommitmentScheme<P>>::scale_comm(comm1, rho1);
        let comm2_rho2 = <Self as CommitmentScheme<P>>::scale_comm(comm2, rho2);

        let comm1_rho1_for_pairing = <Self as CommitmentScheme<P>>::extract_commit_swh(comm1_rho1)?;
        let comm2_rho2_for_pairing = <Self as CommitmentScheme<P>>::extract_commit_swh(comm2_rho2)?;

        // Compute the value of x1 * rho1 in G1
        let mut x1_rho1_in_G1 = ck.g1;
        x1_rho1_in_G1.mul_assign(x1);
        x1_rho1_in_G1.mul_assign(rho1);

        // Compute the value of x2 * rho2 in G1
        let mut x2_rho2_in_G1 = ck.g1;
        x2_rho2_in_G1.mul_assign(x2);
        x2_rho2_in_G1.mul_assign(rho2);

        // Compute w1 * rho1 for pairing2
        let mut prf1_rho1_y1 = prf1;
        prf1_rho1_y1.mul_assign(y1);
        prf1_rho1_y1.mul_assign(rho1);
        // Compute w2 * rho2 for pairing2
        let mut prf2_rho2_y2 = prf2;
        prf2_rho2_y2.mul_assign(y2);
        prf2_rho2_y2.mul_assign(rho2);

        // Compute w1 * rho1 for pairing2
        let mut prf1_rho1 = prf1;
        prf1_rho1.mul_assign(rho1);
        // Compute w2 * rho2 for pairing2
        let mut prf2_rho2 = prf2;
        prf2_rho2.mul_assign(rho2);

        // COMPUTING PAIRING 1
        // LHS
        let lhs1 = comm1_rho1_for_pairing - x1_rho1_in_G1 + prf1_rho1_y1 + comm2_rho2_for_pairing
            - x2_rho2_in_G1
            + prf2_rho2_y2;

        // COMPUTING PAIRING 2
        // LHS
        let lhs2 = -prf1_rho1 - prf2_rho2;

        // COMPUTING THE PAIRINGS
        let mut list_pairings: Vec<(P::G1Prepared, P::G2Prepared)> = Vec::new();
        list_pairings.push((
            P::G1Prepared::from(lhs1.into()),
            P::G2Prepared::from(ck.g2.into()),
        ));
        list_pairings.push((
            P::G1Prepared::from(lhs2.into()),
            P::G2Prepared::from(ck.alpha_g2.into()),
        ));

        let pairing_result = P::product_of_pairings(&list_pairings);
        let result = pairing_result.is_one();

        if result {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }
}

impl<P: PairingEngine, C: CommitmentScheme<P>> CPEvals<P, C> for CS1 {
    fn prove_poly_eval(
        ck: &Self::CS_PP,
        p: &DPolynomial<P::Fr>,
        x: (&P::Fr, &P::Fr), // input to be use as
    ) -> Result<P::G1Projective, CPError> {
        let (i, phi_i) = x;
        let psi = kate_division::<P::Fr>(p, *i, *phi_i);

        let w_i: <P as PairingEngine>::G1Projective =
            eval_on_pnt_in_grp::<P::G1Projective>(&psi, &ck.pws_g1); // from: &pp.pws_in_g1);

        // Find a way to check the point is on the curve
        if true {
            Ok(w_i)
        } else {
            Err(CPError::BadPrfInput)
        }
    }

    // Function for evaluating several polys p1, p2, p3 on the same point and proving there value
    fn multiple_evals_on_same_point(
        ck: &Self::CS_PP,
        a_bi: (P::Fr, Vec<P::Fr>),
        pi: Vec<DPolynomial<P::Fr>>,
        ci: Vec<Self::CommT>,
        rhoi: Vec<P::Fr>,
    ) -> Result<(P::G1Projective, P::Fr, Self::CommT), CPError> {
        // Sketch for two value eval
        // to be scale with for loops later
        let mut bistar: Vec<P::Fr> = Vec::new();
        let mut pistar: Vec<DPolynomial<P::Fr>> = Vec::new();
        let mut cistar: Vec<_> = Vec::new();

        let (a, bi) = a_bi;

        // Using closure is probably better
        // For now iterating on bi, rhoi and pi to construct bistar and pistar
        for (b, c, rho, p) in izip!(bi, ci, rhoi, &pi) {
            bistar.push(b * rho);
            pistar.push(p.mul(&poly_const!(P::Fr, rho)).clone());
            cistar.push(<Self as CommitmentScheme<P>>::scale_comm(c, rho));
        }

        // compute p*(x) = sum(rhoi * pi)
        let p_star = pistar.iter().fold(
            DPolynomial::<P::Fr>::from_coefficients_slice(&[P::Fr::zero()]),
            |acc, p| acc + p.clone(),
        );

        // compute b* = sum(rhoi * bi)
        let b_star = bistar.iter().fold(P::Fr::zero(), |acc, b_val| acc + b_val); // image
                                                                                  // compute c* = sum(rhoi * ci)
        let c_star = cistar
            .into_iter()
            .fold(Self::CommT::Swh(P::G1Projective::zero()), |acc, c| acc + c); // commitments
                                                                                // Computes the proof that each pi evalutes to bi on a
        let proof = <Self as CPEvals<P, C>>::prove_poly_eval(ck, &p_star, (&a, &b_star)).unwrap();

        Ok((proof, b_star, c_star))
    }

    // Function for evaluating several polys p1, p2, p3
    fn multiple_evals(
        ck: &Self::CS_PP,
        ai_bi: (&Vec<P::Fr>, Vec<Vec<P::Fr>>),
        pi: Vec<Vec<DPolynomial<P::Fr>>>,
        ci: Vec<Vec<Self::CommT>>,
        rhoi: Vec<Vec<P::Fr>>,
    ) -> Result<Vec<(P::G1Projective, P::Fr, Self::CommT)>, CPError> //Result<(DPolynomial<P::Fr>,(P::Fr,P::Fr)),CPError>
    {
        let mut proof = Vec::new();
        let (ai, bi) = ai_bi;

        for (i, a) in ai.iter().enumerate() {
            let proof_a = <Self as CPEvals<P, C>>::multiple_evals_on_same_point(
                ck,
                (*a, bi[i].to_vec()),
                pi[i].to_vec(),
                ci[i].to_vec(),
                rhoi[i].to_vec(),
            )
            .unwrap();
            proof.push(proof_a);
        }
        Ok(proof)
    }
}

impl<P: PairingEngine, C: CommitmentScheme<P>> CPEval<P, C> for CS2 {
    fn prove_poly_eval(
        ck: &Self::CS_PP,
        p: &DPolynomial<P::Fr>,
        x: (&P::Fr, &P::Fr), // input to be use as
    ) -> Result<P::G1Projective, CPError> {
        let (i, phi_i) = x;
        let psi = kate_division::<P::Fr>(p, *i, *phi_i);

        let w_i = eval_on_pnt_in_grp::<P::G1Projective>(&psi, &ck.pws_g1); // from: &pp.pws_in_g1);

        // Find a way to check the point is on the curve
        if true {
            Ok(w_i)
        } else {
            Err(CPError::BadPrfInput)
        }
    }

    fn verify_with_eq_check(
        ck: &Self::CS_PP,
        prf: &P::G1Projective,
        comm: Self::CommT,
        a_b: (&P::Fr, &P::Fr),
    ) -> Result<(), CPError> {
        let (a, b) = a_b;

        let commit = <Self as CommitmentScheme<P>>::extract_commit_swh(comm)?;

        // Compute the value of b in G1
        let mut b_in_G1 = ck.g1;
        b_in_G1.mul_assign(*b);

        // Compute the value of c - b in G1
        let lhs_pairing2 = commit - b_in_G1;

        // get the value of a in G2
        let mut a_in_G2 = ck.g2;
        a_in_G2.mul_assign(*a);

        // Compute the value s - a in G2
        let rhs_pairing1 = ck.alpha_g2 - a_in_G2;

        let term1 = P::pairing(*prf, rhs_pairing1);
        let term2 = P::pairing(lhs_pairing2, ck.g2); //P::pairing(c.sub(&ck.g1.mul(b)), ck.alpha_g2);

        if term1.eq(&term2) {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }

    // optimized verify doing two pairings and check for equality to 1
    fn verify(
        ck: &Self::CS_PP,
        prf: &P::G1Projective,
        comm: Self::CommT,
        a_b: (&P::Fr, &P::Fr),
    ) -> Result<(), CPError> {
        let (a, b) = a_b;

        let commit = <Self as CommitmentScheme<P>>::extract_commit_swh(comm)?;

        // Compute the value of b in G1
        let mut b_in_G1 = ck.g1;
        b_in_G1.mul_assign(*b);

        // Compute the value of c - b in G1
        let lhs_pairing2 = b_in_G1 - commit;

        // get the value of a in G2
        let mut a_in_G2 = ck.g2;
        a_in_G2.mul_assign(*a);

        // Compute the value s - a in G2
        let rhs_pairing1 = ck.alpha_g2 - a_in_G2;

        let mut list_pairings: Vec<(P::G1Prepared, P::G2Prepared)> = Vec::new();
        list_pairings.push((
            P::G1Prepared::from((*prf).into()),
            P::G2Prepared::from(rhs_pairing1.into()),
        ));
        list_pairings.push((
            P::G1Prepared::from(lhs_pairing2.into()),
            P::G2Prepared::from(ck.g2.into()),
        ));

        let pairing_result = P::product_of_pairings(&list_pairings);
        let result = pairing_result.is_one();

        if result {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }

    // optimized verify doing two pairings with rng and check for equality to 1
    // Should be removed?
    fn verify_with_rng(
        ck: &Self::CS_PP,
        prf: P::G1Projective,
        comm: Self::CommT,
        a_b: (&P::Fr, &P::Fr),
    ) -> Result<(), CPError> {
        let rng = &mut ark_std::test_rng();
        //  FS RNG FOR SECOND EQ
        let rho = P::Fr::rand(rng);

        let (a, b) = a_b;

        let comm_rho = <Self as CommitmentScheme<P>>::scale_comm(comm, rho);
        let commit_for_pairing = <Self as CommitmentScheme<P>>::extract_commit_swh(comm_rho)?;

        let mut proof_rho = prf;
        proof_rho.mul_assign(rho);

        // Compute the value of b in G1
        let mut b_in_G1 = ck.g1;
        b_in_G1.mul_assign(*b);
        b_in_G1.mul_assign(rho);

        // Compute the value of c - b in G1
        let lhs_pairing2 = commit_for_pairing - b_in_G1;

        // get the value of a in G2
        let mut a_in_G2 = ck.g2;
        a_in_G2.mul_assign(*a);

        // Compute the value s - a in G2
        let rhs_pairing1 = ck.alpha_g2 - a_in_G2;

        let mut list_pairings: Vec<(P::G1Prepared, P::G2Prepared)> = Vec::new();
        list_pairings.push((
            P::G1Prepared::from((-proof_rho).into()),
            P::G2Prepared::from(rhs_pairing1.into()),
        ));
        list_pairings.push((
            P::G1Prepared::from(lhs_pairing2.into()),
            P::G2Prepared::from(ck.g2.into()),
        ));

        let pairing_result = P::product_of_pairings(&list_pairings);

        let result = pairing_result.is_one();

        if result {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }

    // optimized verify doing two pairings with rng and batching proofs and check for equality to 1
    fn batch_verify_with_rng(
        ck: &Self::CS_PP,
        prfs: (P::G1Projective, P::G1Projective),
        comms: (Self::CommT, Self::CommT),
        a_b: ((P::Fr, P::Fr), (P::Fr, P::Fr)), // stands for (a, b) in the pairing describe in the paper
    ) -> Result<(), CPError> {
        // Verifier RNG to make sure the proof is correct
        let rng = &mut ark_std::test_rng();
        let rho1 = P::Fr::rand(rng);
        let rho2 = P::Fr::rand(rng);

        // UNPACKING
        // Here yi is the point and xi the evaluation
        let ((y1, x1), (y2, x2)) = a_b;
        let (prf1, prf2) = prfs;
        let (comm1, comm2) = comms;

        let comm1_rho1 = <Self as CommitmentScheme<P>>::scale_comm(comm1, rho1);
        let comm2_rho2 = <Self as CommitmentScheme<P>>::scale_comm(comm2, rho2);

        let comm1_rho1_for_pairing = <Self as CommitmentScheme<P>>::extract_commit_swh(comm1_rho1)?;
        let comm2_rho2_for_pairing = <Self as CommitmentScheme<P>>::extract_commit_swh(comm2_rho2)?;

        // Compute the value of x1 * rho1 in G1
        let mut x1_rho1_in_G1 = ck.g1;
        x1_rho1_in_G1.mul_assign(x1);
        x1_rho1_in_G1.mul_assign(rho1);

        // Compute the value of x2 * rho2 in G1
        let mut x2_rho2_in_G1 = ck.g1;
        x2_rho2_in_G1.mul_assign(x2);
        x2_rho2_in_G1.mul_assign(rho2);

        // Compute w1 * rho1 for pairing2
        let mut prf1_rho1_y1 = prf1;
        prf1_rho1_y1.mul_assign(y1);
        prf1_rho1_y1.mul_assign(rho1);
        // Compute w2 * rho2 for pairing2
        let mut prf2_rho2_y2 = prf2;
        prf2_rho2_y2.mul_assign(y2);
        prf2_rho2_y2.mul_assign(rho2);

        // Compute w1 * rho1 for pairing2
        let mut prf1_rho1 = prf1;
        prf1_rho1.mul_assign(rho1);
        // Compute w2 * rho2 for pairing2
        let mut prf2_rho2 = prf2;
        prf2_rho2.mul_assign(rho2);

        // COMPUTING PAIRING 1
        // LHS
        let lhs1 = comm1_rho1_for_pairing - x1_rho1_in_G1 + prf1_rho1_y1 + comm2_rho2_for_pairing
            - x2_rho2_in_G1
            + prf2_rho2_y2;

        // COMPUTING PAIRING 2
        // LHS
        let lhs2 = -prf1_rho1 - prf2_rho2;

        // COMPUTING THE PAIRINGS
        let mut list_pairings: Vec<(P::G1Prepared, P::G2Prepared)> = Vec::new();
        list_pairings.push((
            P::G1Prepared::from(lhs1.into()),
            P::G2Prepared::from(ck.g2.into()),
        ));
        list_pairings.push((
            P::G1Prepared::from(lhs2.into()),
            P::G2Prepared::from(ck.alpha_g2.into()),
        ));

        let pairing_result = P::product_of_pairings(&list_pairings);
        let result = pairing_result.is_one();

        if result {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }
}

impl<P: PairingEngine, C: CommitmentScheme<P>> CPqeq<P, C> for CS2 {
    // Check (2) in the Verify QEQ
    // commits_g1 and commits_g2 contain the lists of commits for pairing check
    fn verify_complementary_commits(
        ck: &Self::CS_PP,
        commits_g1: &Vec<Self::CommSwh>,
        commits_g2: &Vec<Self::CommRel>,
    ) -> Result<(), CPError> {
        for (c1, c2) in izip!(commits_g1, commits_g2) {
            let term1 = P::pairing(*c1, ck.g2);
            let term2 = P::pairing(ck.g1, *c2);

            if !term1.eq(&term2) {
                return Err(CPError::BadPrf);
            }
        }

        Ok(())
    }

    // Check (3) in the Verify QEQ
    // commits_g1 and commits_g2 contain the lists of commits to evaluate the Ghat poly
    // the order of the commits should always follow the order of the MV variables
    // i.e. if it has 3 variables in its terms represented as:
    // (1,(1,1)) = X, (1,(2,1)) = Y and (1,(3,1)) =Z
    // then the commit vectors should have the form:
    // commits_g1 = [commit_X_G1, commit_Y_G1, commit_Z_G1]
    // commits_g2 = [commit_X_G2, commit_Y_G2, commit_Z_G2]
    fn verify_Ghat_eval(
        ck: &Self::CS_PP,
        g_hat: SPolynomial<P::Fr, SparseTerm>,
        commits_g1: &Vec<Self::CommSwh>,
        commits_g2: &Vec<Self::CommRel>,
    ) -> Result<(), CPError> {
        let mut list_pairings: Vec<(P::G1Prepared, P::G2Prepared)> = Vec::new(); //[(P::G1Prepared,P::G2Prepared)]
        let mut list_deg1_terms_g1 = Vec::new();
        let mut list_deg1_terms_g2 = Vec::new();

        // Parsing the multivariate poly to separate the terms
        // of degree 2 from the others
        let terms_g_hat = g_hat.terms();

        // TODO check the types of the commits for deg 1 terms
        // For now we assume the deg 1 terms have a g1 commit
        for term in terms_g_hat {
            if term.1.degree() == 2 {
                // case where the term is in the form X * Y where X is in G1 and Y in G2
                if term.1.len() == 2 {
                    // Be careful of the order in which the variables are used as the MV poly orders them internally
                    let prepared_g1_commit: P::G1Prepared = P::G1Prepared::from(
                        commits_g1[term.1.vars()[0]].mul(&term.0.into_repr()).into(),
                    );
                    let prepared_g2_commit: P::G2Prepared =
                        P::G2Prepared::from(commits_g2[term.1.vars()[1]].into());
                    list_pairings.push((prepared_g1_commit, prepared_g2_commit));

                // Case where the term has the form X * X
                } else {
                    let prepared_g1_commit: P::G1Prepared = P::G1Prepared::from(
                        commits_g1[term.1.vars()[0]].mul(&term.0.into_repr()).into(),
                    );
                    let prepared_g2_commit: P::G2Prepared =
                        P::G2Prepared::from(commits_g2[term.1.vars()[0]].into());
                    list_pairings.push((prepared_g1_commit, prepared_g2_commit));
                }

            // Terms in the form X
            } else if term.1.degree() == 1 {
                // For now adds both commitment to the vec
                // Works when the commitments are only in one vector and 0 in the other
                // To be changed at some point
                list_deg1_terms_g1.push(commits_g1[term.1.vars()[0]].mul(&term.0.into_repr()));
                list_deg1_terms_g2.push(commits_g2[term.1.vars()[0]].mul(&term.0.into_repr()));
            }
        }

        let deg1_commits = list_deg1_terms_g1
            .iter()
            .fold(P::G1Projective::zero(), |acc, x| acc + x);
        let deg2_commits = list_deg1_terms_g2
            .iter()
            .fold(P::G2Projective::zero(), |acc, x| acc + x);

        list_pairings.push((
            P::G1Prepared::from(deg1_commits.into()),
            P::G2Prepared::from(ck.g2.into()),
        ));
        list_pairings.push((
            P::G1Prepared::from(ck.g1.into()),
            P::G2Prepared::from(deg2_commits.into()),
        ));

        let pairing_result = P::product_of_pairings(&list_pairings);

        let result = pairing_result.is_one();

        assert_eq!(result, true);

        Ok(())
    }

    // compute the commits in the complementary group for sets of polys
    // TODO: Modify the function to properly take care of the panic!
    fn compute_complementary_commits(
        ck: &Self::CS_PP,
        p_to_g1: Vec<&DPolynomial<P::Fr>>, // poly to commit to g1
        p_to_g2: Vec<&DPolynomial<P::Fr>>, // poly to commit to g2
    ) -> (Vec<Self::CommSwh>, Vec<Self::CommRel>) {
        let comms_g1 = p_to_g1
            .iter()
            .map(|p| {
                let commit;
                match Self::commit_swh(&ck, p) {
                    TypedComm::Swh(c) => {
                        commit = c;
                    }
                    _ => {
                        panic!("Branch should not be reached!");
                    }
                }
                commit
            })
            .collect::<Vec<Self::CommSwh>>();

        let comms_g2 = p_to_g2
            .iter()
            .map(|p| {
                let commit;
                match Self::commit_rel(&ck, p) {
                    TypedComm::Rel(c) => {
                        commit = c;
                    }
                    _ => {
                        panic!("Branch should not be reached!");
                    }
                }
                commit
            })
            .collect::<Vec<Self::CommRel>>();

        (comms_g1, comms_g2)
    }

    fn batch_verify_eq2(
        ck: &Self::CS_PP,

        c_r_prime: Self::CommT,
        c_q_prime: Self::CommT,

        domain_k_size: usize,

        c_cr: Vec<Self::CommT>,
        cr_coeff: Vec<P::Fr>, // sigma_over_K_n2 * xi * yj

        c_cr_prime: Vec<Self::CommT>,
        cr_prime_coeff: Vec<P::Fr>, // n^2 * x^i * y^j

        c_vcrL: Vec<Self::CommT>,
        c_vcrR: Vec<Self::CommT>,
        vcrL_vcrR_coeff: Vec<P::Fr>, // z_h(x) * z_h(y) * x^i * y^j
        alpha: P::Fr,
    ) -> Result<(), CPError> {
        // Verifier RNG to make sure the proof is correct
        let rng = &mut ark_std::test_rng();
        let rho_eq2 = P::Fr::rand(rng);

        let c_r_prime = <Self as CommitmentScheme<P>>::extract_commit_swh(c_r_prime)?;
        let c_q_prime = <Self as CommitmentScheme<P>>::extract_commit_swh(c_q_prime)?;

        let c_cr = <Self as CommitmentScheme<P>>::extract_vec_commit_rel(c_cr)?;

        let c_cr_prime = <Self as CommitmentScheme<P>>::extract_vec_commit_rel(c_cr_prime)?;

        let c_z = ck.pws_g2[domain_k_size] - ck.g2;

        let c_vcrL = <Self as CommitmentScheme<P>>::extract_vec_commit_rel(c_vcrL)?;
        let c_vcrR = <Self as CommitmentScheme<P>>::extract_vec_commit_rel(c_vcrR)?;

        let C_cr: P::G2Projective =
            c_cr.iter()
                .zip(cr_coeff)
                .fold(P::G2Projective::zero(), |acc, (commit, coeff)| {
                    // println!("commit = {:?}", commit);

                    acc + commit.mul(coeff.into_repr())
                });

        let C_cr_prime: P::G2Projective = c_cr_prime
            .iter()
            .zip(cr_prime_coeff)
            .fold(P::G2Projective::zero(), |acc, (commit, coeff)| {
                acc + commit.mul(coeff.into_repr())
            });

        let C_vcrL_vcrR: P::G2Projective = izip!(c_vcrL, c_vcrR, vcrL_vcrR_coeff).fold(
            P::G2Projective::zero(),
            |acc, (commit_vcl, commit_vcr, coeff)| {
                acc + (commit_vcr.mul((alpha * coeff).into_repr())
                    + commit_vcl.mul(coeff.into_repr()))
            },
        );

        let mut list_pairings: Vec<(P::G1Prepared, P::G2Prepared)> = Vec::new();

        list_pairings.push((
            P::G1Prepared::from(c_r_prime.into()),
            P::G2Prepared::from(C_cr_prime.into()),
        ));

        list_pairings.push((
            P::G1Prepared::from(-c_q_prime.into()),
            P::G2Prepared::from(c_z.into()),
        ));

        list_pairings.push((
            P::G1Prepared::from(ck.g1.into()),
            P::G2Prepared::from((C_vcrL_vcrR + C_cr).into()),
        ));

        let pairing_result = P::product_of_pairings(&list_pairings);
        let result = pairing_result.is_one();

        if result {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }
}

impl<P: PairingEngine, C: CommitmentScheme<P>> CPdeg<P, C> for CS2 {
    fn derive(ck: &Self::CS_PP, deg_bounds: &Vec<usize>) -> Result<Vec<P::G2Projective>, CPError> {
        let mut vk_d = Vec::<P::G2Projective>::new();
        for d in deg_bounds.iter() {
            if ck.deg_bound >= *d {
                vk_d.push(ck.pws_g2[ck.deg_bound - d]);
            } else {
                return Err(CPError::BadPrfInput);
            }
        }

        Ok(vk_d)
    }

    fn verify_one_deg_bound(
        ck: &Self::CS_PP,
        vk_d: P::G2Projective, // Not vec because checking one deg bound
        c_star: Self::CommT,
        c_prime: Self::CommT,
    ) -> Result<(), CPError> {
        let c_star = <Self as CommitmentScheme<P>>::extract_commit_swh(c_star)?;
        let c_prime = <Self as CommitmentScheme<P>>::extract_commit_swh(c_prime)?;

        let lhs_p1 = c_prime;
        let rhs_p1 = vk_d;

        // Check if the minus here is correct
        let lhs_p2 = -c_star;
        let rhs_p2 = ck.g2;

        let mut list_pairings: Vec<(P::G1Prepared, P::G2Prepared)> = Vec::new();
        list_pairings.push((
            P::G1Prepared::from(lhs_p1.into()),
            P::G2Prepared::from(rhs_p1.into()),
        ));
        list_pairings.push((
            P::G1Prepared::from(lhs_p2.into()),
            P::G2Prepared::from(rhs_p2.into()),
        ));

        let pairing_result = P::product_of_pairings(&list_pairings);

        let result = pairing_result.is_one();

        if result {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }

    fn verify_deg_bound(
        ck: &Self::CS_PP,
        vk_d: &Vec<P::G2Projective>,
        deg_bounds: &Vec<usize>,
        c_star: Vec<Self::CommT>,
        c_prime: Vec<Self::CommT>,
    ) -> Result<(), CPError> {
        let c_star = <Self as CommitmentScheme<P>>::extract_vec_commit_swh(c_star)?;
        let c_prime = <Self as CommitmentScheme<P>>::extract_vec_commit_swh(c_prime)?;

        let mut list_pairings: Vec<(P::G1Prepared, P::G2Prepared)> = Vec::new();
        for (i, d) in deg_bounds.iter().enumerate() {
            let lhs_p1 = c_prime[i];
            let rhs_p1 = vk_d[i];

            list_pairings.push((
                P::G1Prepared::from(lhs_p1.into()),
                P::G2Prepared::from(rhs_p1.into()),
            ));
        }

        // Adding the batched pairing of the commitments to the r* polys
        let lhs_p2 = -c_star
            .iter()
            .fold(P::G1Projective::zero(), |acc, ci| acc + ci);
        let rhs_p2 = ck.g2;
        list_pairings.push((
            P::G1Prepared::from(lhs_p2.into()),
            P::G2Prepared::from(rhs_p2.into()),
        ));

        let pairing_result = P::product_of_pairings(&list_pairings);

        let result = pairing_result.is_one();

        if result {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }

    // Batch verification for CS2 eq1 and degree bounds
    // performs the check in 4 pairings using rng and batching
    fn batch_verify_deg_bound_eq1(
        ck: &Self::CS_PP,
        vk_d: &Vec<P::G2Projective>,
        deg_bounds: &Vec<usize>,
        c_star: Vec<Self::CommT>,
        c_prime: Vec<Self::CommT>,
        prf: P::G1Projective,
        comm_eq1: Self::CommT,
        a_b: (&P::Fr, &P::Fr), // stands for (a, b) in the pairing describe in the paper
    ) -> Result<(), CPError> {
        // Verifier RNG to make sure the proof is correct
        let rng = &mut ark_std::test_rng();
        let rho_eq1 = P::Fr::rand(rng);
        let rho_deg_r = P::Fr::rand(rng);
        let rho_deg_r_prime = P::Fr::rand(rng);

        let (y, x) = a_b;

        let c_star = <Self as CommitmentScheme<P>>::extract_vec_commit_swh(c_star)?;
        let c_prime = <Self as CommitmentScheme<P>>::extract_vec_commit_swh(c_prime)?;

        let mut list_pairings: Vec<(P::G1Prepared, P::G2Prepared)> = Vec::new();

        // Compute the value of x1 * rho1 in G1
        let mut x_rho1_in_G1 = ck.g1;
        x_rho1_in_G1.mul_assign(*x);
        x_rho1_in_G1.mul_assign(rho_eq1);

        // Compute w1 * rho1 for pairing2
        let mut prf_rho_eq1 = prf;
        prf_rho_eq1.mul_assign(rho_eq1);

        // get the value of a in G2
        let mut y_in_G2 = ck.g2;
        y_in_G2.mul_assign(*y);

        let rhs1_pairing_eq1 = ck.alpha_g2 - y_in_G2;

        list_pairings.push((
            P::G1Prepared::from(prf_rho_eq1.into()),
            P::G2Prepared::from(rhs1_pairing_eq1.into()),
        ));

        let comm_rho_eq1 = <Self as CommitmentScheme<P>>::scale_comm(comm_eq1, rho_eq1);
        let comm_rho_eq1 = <Self as CommitmentScheme<P>>::extract_commit_swh(comm_rho_eq1)?;

        let lhs2_pairing_eq1 = x_rho1_in_G1 - comm_rho_eq1;

        let mut lhs_pairing_deg_r = c_prime[0];
        lhs_pairing_deg_r.mul_assign(rho_deg_r);
        let rhs_pairing_deg_r = vk_d[0];

        list_pairings.push((
            P::G1Prepared::from(lhs_pairing_deg_r.into()),
            P::G2Prepared::from(rhs_pairing_deg_r.into()),
        ));

        let mut lhs_pairing_deg_r_prime = c_prime[1];
        lhs_pairing_deg_r_prime.mul_assign(rho_deg_r_prime);
        let rhs_pairing_deg_r_prime = vk_d[1];

        list_pairings.push((
            P::G1Prepared::from(lhs_pairing_deg_r_prime.into()),
            P::G2Prepared::from(rhs_pairing_deg_r_prime.into()),
        ));

        let mut batch_lhs_pairing_deg_r = c_star[0];
        batch_lhs_pairing_deg_r.mul_assign(rho_deg_r);

        let mut batch_lhs_pairing_deg_r_prime = c_star[1];
        batch_lhs_pairing_deg_r_prime.mul_assign(rho_deg_r_prime);

        // Adding the batched pairing of the commitments to the r* polys
        let lhs_pairing_batch =
            lhs2_pairing_eq1 - batch_lhs_pairing_deg_r - batch_lhs_pairing_deg_r_prime;
        let rhs_pairing_batch = ck.g2;

        list_pairings.push((
            P::G1Prepared::from(lhs_pairing_batch.into()),
            P::G2Prepared::from(rhs_pairing_batch.into()),
        ));

        let pairing_result = P::product_of_pairings(&list_pairings);
        let result = pairing_result.is_one();

        if result {
            return Ok(());
        } else {
            return Err(CPError::BadPrf);
        }
    }

    fn prove_one_deg_bound(
        ck: &Self::CS_PP,
        deg_bound: usize,
        polys: Vec<DPolynomial<P::Fr>>,
        commits: Vec<Self::CommSwh>,
        rho_vec: Vec<P::Fr>,
    ) -> Result<(Self::CommT, Self::CommT), CPError> {
        let mut c_prime_vec = Vec::<P::G1Projective>::new();
        let mut p_prime_vec = Vec::<DPolynomial<P::Fr>>::new();

        for (i, &c) in commits.iter().enumerate() {
            let rhoi = rho_vec[i].clone();
            let poly_rhoi = poly_const!(P::Fr, rhoi);
            let rhoi_pi = &polys[i].mul(&poly_rhoi);

            p_prime_vec.push(rhoi_pi.clone());
            c_prime_vec.push(c.mul(rho_vec[i].into_repr()));
        }

        let c_prime = c_prime_vec
            .iter()
            .fold(P::G1Projective::zero(), |acc, ci| acc + ci);
        let p_prime = p_prime_vec
            .iter()
            .fold(poly_const!(P::Fr, P::Fr::zero()), |acc, p| acc + p.clone());
        let p_star = shift_by_n::<P::Fr>(ck.deg_bound - deg_bound, &p_prime);
        let c_star = Self::commit_swh(&ck, &p_star);

        Ok((c_star, Self::CommT::Swh(c_prime)))
    }

    fn prove_deg_bound(
        ck: &Self::CS_PP,
        deg_bounds: &Vec<usize>,
        polys: Vec<Vec<DPolynomial<P::Fr>>>,
        commits: Vec<Vec<Self::CommT>>,
        rho_vec: Vec<Vec<P::Fr>>,
    ) -> Result<(Vec<Self::CommT>, Vec<Self::CommT>), CPError> {
        let mut c_prime = Vec::<Self::CommT>::new();
        let mut p_prime = Vec::<DPolynomial<P::Fr>>::new();

        let mut p_star = Vec::<DPolynomial<P::Fr>>::new();
        let mut c_star = Vec::<Self::CommT>::new();

        let num_deg_bounds = deg_bounds.len();
        for (j, d) in deg_bounds.iter().enumerate() {
            let mut temp_vec_p_prime = Vec::<DPolynomial<P::Fr>>::new();
            let mut temp_vec_c_prime = Vec::<Self::CommT>::new();

            for (i, &c) in commits[j].iter().enumerate() {
                let rhoi = rho_vec[j][i].clone();
                let poly_rhoi = poly_const!(P::Fr, rhoi);
                let rhoi_pi = &polys[j][i].mul(&poly_rhoi);

                temp_vec_p_prime.push(rhoi_pi.clone());
                temp_vec_c_prime.push(<Self as CommitmentScheme<P>>::scale_comm(c, rho_vec[j][i]));
            }
            c_prime.push(
                temp_vec_c_prime
                    .iter()
                    .fold(Self::CommT::Swh(P::G1Projective::zero()), |acc, &ci| {
                        acc + ci
                    }),
            );
            p_prime.push(
                temp_vec_p_prime
                    .iter()
                    .fold(poly_const!(P::Fr, P::Fr::zero()), |acc, p| acc + p.clone()),
            );
            p_star.push(shift_by_n::<P::Fr>(ck.deg_bound - d, &p_prime[j]));

            c_star.push(Self::commit_swh(&ck, &p_star[j]));
        }

        Ok((c_star, c_prime))
    }
}
