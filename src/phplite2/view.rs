#![macro_use]

use crate::matrixutils::*;
use crate::php::*;
use ark_ff::{FftField, Field};
use ark_poly::univariate::DensePolynomial as DPolynomial;

use ark_ec::PairingEngine;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};

use ark_std::io::{Read, Write};

use crate::comms::*;
use dict::{Dict, DictIface};

// TODO: move somewhere else
fn append_to_dict<T: Clone>(d: &mut Dict<T>, to_append_d: &Dict<T>) {
    to_append_d.iter().for_each(|o| {
        d.add(o.key.to_string(), o.val.clone());
    });
}

/* A type used to represent the views of the verifier and prover.
Notice that the types are the same but the views are not necessarily the same
(e.g. priv_vals may contain values for the prover not in the verifier's view)
*/
pub struct View<F: Field> {
    pub priv_vals: Dict<F>,           // private values, e.g. for ZK
    pub idx_oracles: Dict<Oracle<F>>, // offline oracles
    pub prv_oracles: Dict<Oracle<F>>, // online oracles
    pub scalars: Dict<F>,             // this could be a scalar from P or a challenge from V
}

impl<F: Field> View<F> {
    pub fn new() -> View<F> {
        View::<F> {
            priv_vals: Dict::<F>::new(),
            idx_oracles: Dict::<Oracle<F>>::new(),
            prv_oracles: Dict::<Oracle<F>>::new(),
            scalars: Dict::<F>::new(),
        }
    }
    pub fn add_priv_val(&mut self, key: &str, val: F) {
        self.priv_vals.add(key.to_string(), val);
    }

    pub fn add_idx_oracle(&mut self, key: &str, val: Oracle<F>) {
        self.idx_oracles.add(key.to_string(), val);
    }

    pub fn add_prv_oracle(&mut self, key: &str, val: Oracle<F>) {
        self.prv_oracles.add(key.to_string(), val);
    }

    pub fn get_idx_oracle(&self, key: &str) -> Result<DPolynomial<F>, OracleError> {
        self.idx_oracles.get(key).unwrap().clone().get_poly()
    }

    pub fn get_idx_oracle_matrix(&self, key: &str) -> Result<DMatrix<DPolynomial<F>>, OracleError> {
        self.idx_oracles.get(key).unwrap().clone().get_matrix()
    }

    // why do we need a mut ref of self? trying without
    pub fn get_prv_oracle(&self, key: &str) -> Result<DPolynomial<F>, OracleError> {
        self.prv_oracles.get(key).unwrap().clone().get_poly()
    }

    pub fn get_scalar(&mut self, key: &str) -> Result<F, ()> {
        Ok(self.scalars.get(key).unwrap().clone())
    }

    pub fn add_scalar(&mut self, key: &str, val: F) {
        self.scalars.add(key.to_string(), val);
    }

    pub fn append_view(&mut self, to_append: View<F>) {
        append_to_dict(&mut self.priv_vals, &to_append.priv_vals);
        append_to_dict(&mut self.idx_oracles, &to_append.idx_oracles);
        append_to_dict(&mut self.prv_oracles, &to_append.prv_oracles);
        append_to_dict(&mut self.scalars, &to_append.scalars);
    }
}
#[macro_export]
macro_rules! add_to_views {
    (oracle, $pview:ident,$vview:ident,$o:ident) => {
        $pview.add_prv_oracle(stringify!($o), $o.clone().into());
        $vview.add_prv_oracle(stringify!($o), $o.clone().into());
    };
    (scalar, $pview:ident,$vview:ident,$o:ident) => {
        $pview.add_scalar(stringify!($o), $o.clone().into());
        $vview.add_scalar(stringify!($o), $o.clone().into());
    };
    (oracle, $pview:ident,$o:ident) => {
        $pview.add_prv_oracle(stringify!($o), $o.clone().into());
    };
    (scalar, $pview:ident,$o:ident) => {
        $pview.add_scalar(stringify!($o), $o.clone().into());
    };
    // to remove when from is impl for CommOracle
    (commit, $pview:ident,$o:ident) => {
        $pview.add_prv_oracle(stringify!($o), CommOracle::COPoly($o.clone()));
    };
}

// Commitment View
pub struct CView<P: PairingEngine, C: CommitmentScheme<P>> {
    pub idx_oracles: Dict<CommOracle<P, C>>, // offline oracles
    pub prv_oracles: Dict<CommOracle<P, C>>, // online oracles
}

impl<P: PairingEngine, C: CommitmentScheme<P>> CView<P, C> {
    pub fn new() -> CView<P, C> {
        CView::<P, C> {
            idx_oracles: Dict::<CommOracle<P, C>>::new(),
            prv_oracles: Dict::<CommOracle<P, C>>::new(),
        }
    }

    pub fn add_idx_oracle(&mut self, key: &str, val: CommOracle<P, C>) {
        self.idx_oracles.add(key.to_string(), val);
    }

    pub fn add_prv_oracle(&mut self, key: &str, val: CommOracle<P, C>) {
        self.prv_oracles.add(key.to_string(), val);
    }

    // why do we need a mut ref of self? trying without
    pub fn get_prv_oracle(&self, key: &str) -> Result<C::CommT, ()> {
        if let CommOracle::<P, C>::COPoly(c) = self.prv_oracles.get(key).unwrap().clone() {
            Ok(c)
        } else {
            Err(())
        }
    }

    pub fn get_idx_oracle(&self, key: &str) -> Result<C::CommT, ()> {
        if let CommOracle::<P, C>::COPoly(c) = self.idx_oracles.get(key).unwrap().clone() {
            Ok(c)
        } else {
            Err(())
        }
    }

    pub fn get_idx_oracle_matrix(&self, key: &str) -> Result<Vec<Vec<C::CommT>>, ()> {
        if let CommOracle::<P, C>::COMtxPoly(c) = self.idx_oracles.get(key).unwrap().clone() {
            Ok(c)
        } else {
            Err(())
        }
    }

    pub fn append_view(&mut self, to_append: CView<P, C>) {
        append_to_dict(&mut self.idx_oracles, &to_append.idx_oracles);
        append_to_dict(&mut self.prv_oracles, &to_append.prv_oracles);
    }

    pub fn append_prv_comm_oracles(&mut self, comms: &Dict<CommOracle<P, C>>) {
        //append_to_dict(&mut self., to_append.idx_oracles);
        append_to_dict(&mut self.prv_oracles, comms);
    }

    pub fn append_idx_comm_oracles(&mut self, comms: &Dict<CommOracle<P, C>>) {
        //append_to_dict(&mut self.idx_oracles, to_append.idx_oracles);
        append_to_dict(&mut self.idx_oracles, comms);
    }
}

pub fn commit_oracles_dict_swh<P: PairingEngine, C: CommitmentScheme<P>>(
    cm_pp: &C::CS_PP,
    oracles: &Dict<Oracle<P::Fr>>,
) -> Dict<CommOracle<P, C>> {
    let mut comms = Dict::<CommOracle<P, C>>::new();
    for o in oracles {
        let c: CommOracle<P, C> = o.val.commit_swh::<P, C>(cm_pp);
        comms.add(o.key.clone(), c);
    }
    comms
}

pub fn commit_oracles_dict_rel<P: PairingEngine, C: CommitmentScheme<P>>(
    cm_pp: &C::CS_PP,
    oracles: &Dict<Oracle<P::Fr>>,
) -> Dict<CommOracle<P, C>> {
    let mut comms = Dict::<CommOracle<P, C>>::new();
    for o in oracles {
        let c: CommOracle<P, C> = o.val.commit_rel::<P, C>(cm_pp);
        comms.add(o.key.clone(), c);
    }
    comms
}
