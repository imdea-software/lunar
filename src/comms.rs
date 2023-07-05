#![allow(non_camel_case_types)]
use crate::matrixutils::{eval_on_pnt_in_grp, fmsm, gen_pows, kate_division};
use ark_ec::msm::VariableBaseMSM;
use ark_ec::{PairingEngine, ProjectiveCurve};
use ark_ff::{FromBytes, ToBytes};
use ark_ff::{PrimeField, UniformRand, Zero};
use ark_poly::univariate::DensePolynomial as DPolynomial;
use ark_std::fmt::Debug;
use ark_std::rand::RngCore;
use std::ops::MulAssign;

use std::ops::{Add, Mul};
use std::time::Instant;

#[derive(Eq, PartialEq, Clone, Debug, Copy)]
pub enum TypedComm<R: Eq + Clone + Debug, S: Eq + Clone + Debug> {
    Rel(R),
    Swh(S),
}

impl<R: Eq + Clone + Add<Output = R> + Debug, S: Eq + Clone + Add<Output = S> + Debug> Add
    for TypedComm<R, S>
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut bad_input = false;
        if let TypedComm::<R, S>::Rel(r1) = self.clone() {
            if let TypedComm::<R, S>::Rel(r2) = other {
                return TypedComm::<R, S>::Rel(r1 + r2);
            } else {
                bad_input = true;
            }
        } else if let TypedComm::<R, S>::Swh(s1) = self.clone() {
            if let TypedComm::<R, S>::Swh(s2) = other {
                return TypedComm::<R, S>::Swh(s1 + s2);
            } else {
                bad_input = true;
            }
        }
        if bad_input {
            panic!("Commitments should be of the same type");
        } else {
            self.clone() // never reached
        }
    }
}

pub trait CommitmentScheme<P: PairingEngine>
where
    Self: Clone,
{
    type CS_PP: Debug;
    type CommRel: Eq + Clone + Zero + ToBytes + Debug;
    type CommSwh: Eq + Clone + Zero + ToBytes + Debug;
    type CommT: Eq + Clone + Add<Output = Self::CommT> + Debug;

    fn setup<R: rand_core::RngCore>(rng: &mut R, t: usize) -> Self::CS_PP;

    // Functions for commiting to Rel and Swh polynomials
    fn commit_swh(ck: &Self::CS_PP, p: &DPolynomial<P::Fr>) -> Self::CommT;
    fn commit_rel(ck: &Self::CS_PP, p: &DPolynomial<P::Fr>) -> Self::CommT;
    fn ver_com(
        typedC: Self::CommT,
        ck: &Self::CS_PP,
        p: &DPolynomial<P::Fr>,
    ) -> Result<(), CPError>;

    // Functions for commiting to Rel and Swh vectors of polynomials
    fn compute_vec_commits_swh(
        ck: &Self::CS_PP,
        vec_poly: Vec<&DPolynomial<P::Fr>>,
    ) -> Vec<Self::CommT>;
    fn compute_vec_commits_rel(
        ck: &Self::CS_PP,
        vec_poly: Vec<&DPolynomial<P::Fr>>,
    ) -> Vec<Self::CommT>;

    // Extract the inner commitment for easier manipulation
    fn extract_commit_swh(commit_vec: Self::CommT) -> Result<Self::CommSwh, CPError>;
    fn extract_commit_rel(commit_vec: Self::CommT) -> Result<Self::CommRel, CPError>;
    fn extract_vec_commit_swh(commit_vec: Vec<Self::CommT>) -> Result<Vec<Self::CommSwh>, CPError>;
    fn extract_vec_commit_rel(commit_vec: Vec<Self::CommT>) -> Result<Vec<Self::CommRel>, CPError>;

    fn convert_from_rel_to_swh(commit_rel: Self::CommT) -> Self::CommT;

    // Creates a "commitment" to a constant
    fn constant_commit(cs_pp: &Self::CS_PP, constant_term: P::Fr) -> Self::CommT;

    // Convert constant values into commitment for easier computations
    fn convert_const_commT_swh(pt: P::G1Projective) -> Self::CommT;
    fn convert_const_commT_rel(pt: P::G2Projective) -> Self::CommT;

    // some extra functions
    fn scale_comm(c: Self::CommT, f: P::Fr) -> Self::CommT;
}

#[derive(Debug)]
pub enum CPError {
    BadPrfInput,
    BadPrf,
    BadCommit,
    WrongType,
}

/* ====== CS1 ====== */

#[derive(Clone, Debug)]
pub struct CS1;

#[derive(Clone, Debug)]
pub struct CS1_PP<P: PairingEngine> {
    // putting all fields as public for now. May need to be changed later
    pub deg_bound: usize,
    pub g1: P::G1Projective,
    pub g2: P::G2Projective,
    pub pws_g1: Vec<P::G1Projective>,
    pub alpha_g2: P::G2Projective,
}

impl<P: PairingEngine> CS1_PP<P> {
    pub fn new(
        deg_bound: usize,
        g1: P::G1Projective,
        g2: P::G2Projective,
        pws_g1: Vec<P::G1Projective>,
        alpha_g2: P::G2Projective,
    ) -> Self {
        Self {
            deg_bound,
            g1,
            g2,
            pws_g1,
            alpha_g2,
        }
    }
}

macro_rules! scale_comm_impl {
    () => {
        fn scale_comm(c: Self::CommT, f: P::Fr) -> Self::CommT {
            match c {
                Self::CommT::Swh(mut s) => {
                    s.mul_assign(f);
                    return Self::CommT::Swh(s);
                }
                Self::CommT::Rel(mut s) => {
                    s.mul_assign(f);
                    return Self::CommT::Rel(s);
                }
            }
        }
    };
}

macro_rules! vec_commits_swh_impl {
    () => {
        // compute the swh typed commitments of a vec of polys
        fn compute_vec_commits_swh(
            ck: &Self::CS_PP,
            vec_poly: Vec<&DPolynomial<P::Fr>>,
        ) -> Vec<Self::CommT> {
            vec_poly
                .iter()
                .map(|p| Self::commit_swh(&ck, p))
                .collect::<Vec<Self::CommT>>()
        }
    };
}

macro_rules! vec_commits_rel_impl {
    () => {
        // compute the rel typed commitments of a vec of polys
        fn compute_vec_commits_rel(
            ck: &Self::CS_PP,
            vec_poly: Vec<&DPolynomial<P::Fr>>,
        ) -> Vec<Self::CommT> {
            vec_poly
                .iter()
                .map(|p| Self::commit_rel(&ck, p))
                .collect::<Vec<Self::CommT>>()
        }
    };
}

macro_rules! constant_commit_impl {
    () => {
        // compute the swh typed commitments of a scalar
        // Used in the precomputations to facilitate the commitment computations
        fn constant_commit(cs_pp: &Self::CS_PP, constant_term: P::Fr) -> Self::CommT {
            Self::CommT::Swh(cs_pp.g1.mul(constant_term.into_repr()))
        }
    };
}

macro_rules! extract_swh_commit_impl {
    () => {
        // Extract the group element corresponding to the swh commitment
        fn extract_commit_swh(comm: Self::CommT) -> Result<Self::CommSwh, CPError> {
            let commit = match comm {
                Self::CommT::Rel(_) => {
                    return Err(CPError::WrongType);
                } // never reached
                Self::CommT::Swh(c) => c,
            };
            Ok(commit)
        }
    };
}

macro_rules! extract_rel_commit_impl {
    () => {
        // Extract the group element corresponding to the rel commitment
        fn extract_commit_rel(comm: Self::CommT) -> Result<Self::CommRel, CPError> {
            let commit = match comm {
                Self::CommT::Rel(c) => c,
                Self::CommT::Swh(_) => {
                    return Err(CPError::WrongType);
                } // never reached
            };
            Ok(commit)
        }
    };
}

macro_rules! extract_vec_swh_commit_impl {
    () => {
        // Extract the group elements corresponding to a vec of swh commitments
        fn extract_vec_commit_swh(
            commit_vec: Vec<Self::CommT>,
        ) -> Result<Vec<Self::CommSwh>, CPError> {
            let mut vec_comm_to_byte = Vec::new();
            for comm in commit_vec.into_iter() {
                let commit = match &comm {
                    Self::CommT::Rel(_) => {
                        return Err(CPError::WrongType);
                    } // never reached
                    Self::CommT::Swh(c) => *c,
                };
                vec_comm_to_byte.push(commit);
            }
            Ok(vec_comm_to_byte)
        }
    };
}

macro_rules! extract_vec_rel_commit_impl {
    () => {
        // Extract the group elements corresponding to a vec of rel commitments
        fn extract_vec_commit_rel(
            commit_vec: Vec<Self::CommT>,
        ) -> Result<Vec<Self::CommRel>, CPError> {
            let mut vec_comm_to_byte = Vec::new();
            for comm in commit_vec.into_iter() {
                let commit = match &comm {
                    Self::CommT::Rel(c) => *c,
                    Self::CommT::Swh(_) => {
                        return Err(CPError::WrongType);
                    } // never reached
                };
                vec_comm_to_byte.push(commit);
            }
            Ok(vec_comm_to_byte)
        }
    };
}

impl<P: PairingEngine> CommitmentScheme<P> for CS1 {
    type CS_PP = CS1_PP<P>;
    type CommRel = P::G1Projective;
    type CommSwh = P::G1Projective;
    type CommT = TypedComm<Self::CommRel, Self::CommSwh>;

    fn setup<R: rand_core::RngCore>(rng: &mut R, t: usize) -> Self::CS_PP {
        let alpha = P::Fr::rand(rng);
        let powers_of_alpha = gen_pows(alpha, t);
        let g1 = P::G1Projective::rand(rng);
        let pws_g1 = fmsm::<P::G1Projective>(&g1, &powers_of_alpha, t);
        let g2 = P::G2Projective::rand(rng);
        let mut alpha_g2 = g2.clone();
        alpha_g2.mul_assign(alpha);

        Self::CS_PP::new(t, g1, g2, pws_g1, alpha_g2)
    }

    fn commit_swh(
        ck: &Self::CS_PP,
        p: &DPolynomial<P::Fr>,
    ) -> TypedComm<P::G1Projective, P::G1Projective> {
        Self::CommT::Swh(eval_on_pnt_in_grp(&p, &ck.pws_g1))
    }

    fn commit_rel(
        ck: &Self::CS_PP,
        p: &DPolynomial<P::Fr>,
    ) -> TypedComm<P::G1Projective, P::G1Projective> {
        Self::CommT::Rel(eval_on_pnt_in_grp(&p, &ck.pws_g1))
    }

    fn ver_com(
        typedC: Self::CommT,
        ck: &Self::CS_PP,
        p: &DPolynomial<P::Fr>,
    ) -> Result<(), CPError> {
        let mut result = Err(CPError::BadCommit);
        match typedC {
            Self::CommT::Rel(_) => {
                let c_prime = Self::commit_rel(ck, p); // as TypedComm<Self::CommRel,Self::CommSwh>;
                if c_prime == typedC {
                    result = Ok(())
                };
            }
            Self::CommT::Swh(_) => {
                let c_prime = Self::commit_swh(ck, p);
                if c_prime == typedC {
                    result = Ok(())
                };
            }
        };
        result
    }

    fn convert_from_rel_to_swh(commit_rel: Self::CommT) -> Self::CommT {
        let commit = match commit_rel {
            Self::CommT::Rel(c) => c, // never reached
            Self::CommT::Swh(_) => Self::CommSwh::zero(),
        };
        Self::CommT::Swh(commit)
    }

    fn convert_const_commT_swh(pt: P::G1Projective) -> Self::CommT {
        Self::CommT::Swh(pt)
    }

    fn convert_const_commT_rel(pt: P::G2Projective) -> Self::CommT {
        unimplemented!()
    }

    vec_commits_swh_impl!();
    vec_commits_rel_impl!();

    extract_swh_commit_impl!();
    extract_rel_commit_impl!();

    extract_vec_swh_commit_impl!();
    extract_vec_rel_commit_impl!();

    scale_comm_impl!();
    // eqn_prove_impl!();

    constant_commit_impl!();
}

/* ====== CS2 ====== */
#[derive(Clone)]
pub struct CS2;

// Public parameters for KGZ
#[derive(Clone, Debug)]
pub struct CS2_PP<P: PairingEngine> {
    // putting all fields as public. May need to be changed later
    pub deg_bound: usize,
    pub g1: P::G1Projective,
    pub g2: P::G2Projective,
    pub pws_g1: Vec<P::G1Projective>,
    pub pws_g2: Vec<P::G2Projective>,
    pub alpha_g2: P::G2Projective,
}

impl<P: PairingEngine> CS2_PP<P> {
    pub fn new(
        deg_bound: usize,
        g1: P::G1Projective,
        g2: P::G2Projective,
        pws_g1: Vec<P::G1Projective>,
        pws_g2: Vec<P::G2Projective>,
        alpha_g2: P::G2Projective,
    ) -> Self {
        Self {
            deg_bound,
            g1,
            g2,
            pws_g1,
            pws_g2,
            alpha_g2,
        }
    }
}

impl<P: PairingEngine> CommitmentScheme<P> for CS2 {
    type CS_PP = CS2_PP<P>;
    type CommSwh = P::G1Projective;
    type CommRel = P::G2Projective;
    type CommT = TypedComm<Self::CommRel, Self::CommSwh>;

    fn setup<R: rand_core::RngCore>(rng: &mut R, t: usize) -> Self::CS_PP {
        let alpha = P::Fr::rand(rng);
        let powers_of_alpha = gen_pows(alpha, t);
        let g1 = P::G1Projective::rand(rng);
        let pws_g1 = fmsm::<P::G1Projective>(&g1, &powers_of_alpha, t);
        let g2 = P::G2Projective::rand(rng);
        let pws_g2 = fmsm::<P::G2Projective>(&g2, &powers_of_alpha, t);
        let mut alpha_g2 = g2.clone();
        alpha_g2.mul_assign(alpha);

        Self::CS_PP::new(t, g1, g2, pws_g1, pws_g2, alpha_g2)
    }

    fn commit_swh(ck: &Self::CS_PP, p: &DPolynomial<P::Fr>) -> Self::CommT {
        //TypedComm<Self::CommRel, Self::CommSwh> {
        TypedComm::<Self::CommRel, Self::CommSwh>::Swh(eval_on_pnt_in_grp(&p, &ck.pws_g1))
    }

    fn commit_rel(ck: &Self::CS_PP, p: &DPolynomial<P::Fr>) -> Self::CommT {
        //TypedComm<Self::CommRel, Self::CommSwh> {
        TypedComm::<Self::CommRel, Self::CommSwh>::Rel(eval_on_pnt_in_grp(&p, &ck.pws_g2))
    }

    fn ver_com(
        typedC: Self::CommT,
        ck: &Self::CS_PP,
        p: &DPolynomial<P::Fr>,
    ) -> Result<(), CPError> {
        let mut result = Err(CPError::BadCommit);
        match typedC {
            TypedComm::Rel(_) => {
                let c_prime = Self::commit_rel(ck, p) as TypedComm<Self::CommRel, Self::CommSwh>;
                if c_prime == typedC {
                    result = Ok(())
                };
            }
            TypedComm::Swh(_) => {
                let c_prime = Self::commit_swh(ck, p);
                if c_prime == typedC {
                    result = Ok(())
                };
            }
        };
        result
    }

    fn convert_from_rel_to_swh(commit_rel: Self::CommT) -> Self::CommT {
        let commit = match commit_rel {
            Self::CommT::Rel(c) => c, // never reached
            Self::CommT::Swh(_) => Self::CommRel::zero(),
        };
        Self::CommT::Rel(commit)
    }

    fn convert_const_commT_swh(pt: P::G1Projective) -> Self::CommT {
        Self::CommT::Swh(pt)
    }

    fn convert_const_commT_rel(pt: P::G2Projective) -> Self::CommT {
        Self::CommT::Rel(pt)
    }

    vec_commits_swh_impl!();
    vec_commits_rel_impl!();

    extract_swh_commit_impl!();
    extract_rel_commit_impl!();

    extract_vec_swh_commit_impl!();
    extract_vec_rel_commit_impl!();

    scale_comm_impl!();

    constant_commit_impl!();
}
