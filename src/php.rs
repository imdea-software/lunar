use ark_ff::{FftField, Field};
use ark_poly::univariate::DensePolynomial as DPolynomial;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;

use crate::comms::*;
use crate::matrixutils::*;
use ark_ec::PairingEngine;

#[derive(Debug)]
pub enum OracleError {
    NoMatrix,
    NoPoly,
    NoMatrixComm,
    NoPolyComm,
}

// An oracle could simply single poly or a polynomial
#[derive(Clone)]
pub enum Oracle<F: Field> {
    OPoly(DPolynomial<F>),
    OMtxPoly(DMatrix<DPolynomial<F>>),
}

impl<F: Field> From<DPolynomial<F>> for Oracle<F> {
    fn from(p: DPolynomial<F>) -> Self {
        Oracle::<F>::OPoly(p)
    }
}

impl<F: Field> From<DMatrix<DPolynomial<F>>> for Oracle<F> {
    fn from(mtx: DMatrix<DPolynomial<F>>) -> Self {
        Oracle::<F>::OMtxPoly(mtx)
    }
}

impl<F: Field> Oracle<F> {
    pub fn get_matrix(self) -> Result<DMatrix<DPolynomial<F>>, OracleError> {
        if let Oracle::OMtxPoly(matrix) = self {
            Ok(matrix)
        } else {
            Err(OracleError::NoMatrix)
        }
    }

    pub fn get_poly(self) -> Result<DPolynomial<F>, OracleError> {
        if let Oracle::OPoly(poly) = self {
            Ok(poly)
        } else {
            Err(OracleError::NoPoly)
        }
    }
}

impl<F: Field> Oracle<F> {
    pub fn commit_swh<P: PairingEngine<Fr = F>, C: CommitmentScheme<P>>(
        &self,
        cm_pp: &C::CS_PP,
    ) -> CommOracle<P, C> {
        match self {
            Self::OPoly(p) => {
                let c = C::commit_swh(cm_pp, p);
                CommOracle::<P, C>::COPoly(c)
            }
            Self::OMtxPoly(p) => {
                // TODO for Dmtx
                let cmatrix = p
                    .iter()
                    .map(|v| {
                        v.iter()
                            .map(|poly| {
                                let c = C::commit_swh(cm_pp, poly);
                                c
                            })
                            .collect()
                    })
                    .collect();
                CommOracle::<P, C>::COMtxPoly(cmatrix)
            }
        }
    }

    pub fn commit_rel<P: PairingEngine<Fr = F>, C: CommitmentScheme<P>>(
        &self,
        cm_pp: &C::CS_PP,
    ) -> CommOracle<P, C> {
        match self {
            Self::OPoly(p) => {
                let c = C::commit_rel(cm_pp, p);
                CommOracle::<P, C>::COPoly(c)
            }
            Self::OMtxPoly(p) => {
                // TODO for Dmtx
                let cmatrix = p
                    .iter()
                    .map(|v| {
                        v.iter()
                            .map(|poly| {
                                let c = C::commit_rel(cm_pp, poly);
                                c
                            })
                            .collect()
                    })
                    .collect();
                CommOracle::<P, C>::COMtxPoly(cmatrix)
            }
        }
    }
}


#[derive(Clone, Debug)]
pub enum CommOracle<P: PairingEngine, C: CommitmentScheme<P>>
where
    C::CommT: Clone,
{
    COPoly(C::CommT),
    COMtxPoly(DMatrix<C::CommT>),
}

impl<P: PairingEngine, C: CommitmentScheme<P>> CommOracle<P, C> {
    pub fn get_matrix(self) -> Result<DMatrix<C::CommT>, OracleError> {
        if let CommOracle::COMtxPoly(matrix) = self {
            Ok(matrix)
        } else {
            Err(OracleError::NoMatrixComm)
        }
    }

    pub fn get_poly(self) -> Result<C::CommT, OracleError> {
        if let CommOracle::COPoly(poly) = self {
            Ok(poly)
        } else {
            Err(OracleError::NoPolyComm)
        }
    }
}

pub trait Encoder<F: Field> {
    // Constraint System type
    type CS;

    // an encoder returns a vector of oracles
    fn setup(cs: &Self::CS) -> Vec<Oracle<F>>;
}

/// Parameters of a PHPlite instance
#[derive(Clone)]
pub struct CSLiteParams<F: FftField> {
    n: usize,
    m: usize,
    l: usize,
    L: Matrix<F>,
    R: Matrix<F>,
}

impl<F: FftField> CSLiteParams<F> {
    /// Creates a new PHPlite instance with the size parameters n, m, l
    /// and the matrices L and R
    pub fn new(n: usize, m: usize, l: usize, L: Matrix<F>, R: Matrix<F>) -> Self {
        Self { n, m, l, L, R }
    }

    /// Returns the L matrix of the PHPlite instance
    pub fn getL(&self) -> Matrix<F> {
        self.L.clone()
    }

    /// Returns the R matrix of the PHPlite instance
    pub fn getR(&self) -> Matrix<F> {
        self.R.clone()
    }

    /// Returns the size parameters of the PHPlite instance
    pub fn get_params(&self) -> (usize, usize, usize) {
        (self.n, self.m, self.l)
    }
}
