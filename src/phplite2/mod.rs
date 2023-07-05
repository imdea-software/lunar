use crate::php::{CSLiteParams, Encoder, Oracle};

use crate::matrixutils::*;
use crate::{poly_const, x};
use ark_ff::{FftField, Field};
use ark_poly::{
    polynomial::Polynomial, univariate::DensePolynomial as DPolynomial, Evaluations, UVPolynomial,
};

pub mod view;
pub use view::*;
pub mod prover;
pub use prover::*;
pub mod verifier;
pub use verifier::*;

pub struct R1CSLite2;

impl<F: FftField> Encoder<F> for R1CSLite2 {
    type CS = CSLiteParams<F>;
    /// Returns the oracle matrices vcr_L, vcr_R, cr and cr prime
    fn setup(cs: &Self::CS) -> Vec<Oracle<F>> {
        let encodingL = cs.getL().encode_matrix();
        let encodingR = cs.getR().encode_matrix();

        let vcrL = aux_vcr(&encodingL, &encodingR);
        let vcrR = aux_vcr(&encodingR, &encodingL);
        let (cr, cr_p) = aux_cr(&encodingR, &encodingL);
        vec![
            Oracle::OMtxPoly(vcrL),
            Oracle::OMtxPoly(vcrR),
            Oracle::OMtxPoly(cr),
            Oracle::OMtxPoly(cr_p),
        ]
    }
}

pub struct R1CSLite2x;

impl<F: FftField> Encoder<F> for R1CSLite2x {
    type CS = CSLiteParams<F>;
    /// Returns the oracle matrices vcr_L, vcr_R, cr and cr prime
    fn setup(cs: &Self::CS) -> Vec<Oracle<F>> {
        let encodingL = cs.getL().encode_matrix();
        let encodingR = cs.getR().encode_matrix();

        let vcrL = aux_vcr(&encodingL, &encodingR);
        let vcrR = aux_vcr(&encodingR, &encodingL);
        let (cr, _) = aux_cr(&encodingR, &encodingL);
        vec![
            Oracle::OMtxPoly(vcrL),
            Oracle::OMtxPoly(vcrR),
            Oracle::OMtxPoly(cr),
        ]
    }
}

/// Outputs two 3x3 matrices with the oracle polynomials cr, and cr' respectively
pub fn aux_cr<F: FftField>(
    L: &MatrixEncoding<F>,
    R: &MatrixEncoding<F>,
) -> (DMatrix<DPolynomial<F>>, DMatrix<DPolynomial<F>>) {
    let row_L = &L.row_evals.evals;
    let col_L = &L.col_evals.evals;
    let row_R = &R.row_evals.evals;
    let col_R = &R.col_evals.evals;

    let domain_k = L.domain_k;

    let row_prod: Vec<F> = row_L
        .iter()
        .zip(row_R.iter())
        .map(|(&rL, &rR)| rL * rR)
        .collect();

    let row_sum: Vec<F> = row_L
        .iter()
        .zip(row_R.iter())
        .map(|(&rL, &rR)| rL + rR)
        .collect();

    let col_prod: Vec<F> = col_L
        .iter()
        .zip(col_R.iter())
        .map(|(&cL, &cR)| cL * cR)
        .collect();

    let col_sum: Vec<F> = col_L
        .iter()
        .zip(col_R.iter())
        .map(|(&cL, &cR)| cL + cR)
        .collect();

    let cr_00 = row_prod
        .iter()
        .zip(col_prod.iter())
        .map(|(&rp, &cp)| cp * rp)
        .collect();
    let cr_00_poly = Evaluations::from_vec_and_domain(cr_00, domain_k).interpolate();

    let cr_01 = col_sum
        .iter()
        .zip(row_prod.iter())
        .map(|(&cs, &rp)| cs * rp)
        .collect();
    let cr_01_poly = Evaluations::from_vec_and_domain(cr_01, domain_k).interpolate();

    let cr_02_poly = Evaluations::from_vec_and_domain(row_prod, domain_k).interpolate();

    let cr_10 = row_sum
        .iter()
        .zip(col_prod.iter())
        .map(|(&rs, &cp)| rs * cp)
        .collect();
    let cr_10_poly = Evaluations::from_vec_and_domain(cr_10, domain_k).interpolate();

    let cr_11 = row_sum
        .iter()
        .zip(col_sum.iter())
        .map(|(&rs, &cs)| rs * cs)
        .collect();
    let cr_11_poly = Evaluations::from_vec_and_domain(cr_11, domain_k).interpolate();

    let cr_12_poly = Evaluations::from_vec_and_domain(row_sum, domain_k).interpolate();

    let cr_20_poly = Evaluations::from_vec_and_domain(col_prod, domain_k).interpolate();

    let cr_21_poly = Evaluations::from_vec_and_domain(col_sum, domain_k).interpolate();

    let c = vec![
        vec![cr_00_poly, -cr_01_poly, cr_02_poly],
        vec![-cr_10_poly, cr_11_poly, -cr_12_poly],
        vec![cr_20_poly, -cr_21_poly, poly_const!(F, F::one())],
    ];

    let c_prime = c
        .iter()
        .map(|row| row.iter().map(|elem| shift_by_n(1, elem)).collect())
        .collect();
    (c, c_prime)
}

/// Generates the 2x2 matrix used in PHPlite. aux_vcr::<*field*>(L,R) it generates vcr_L
/// and aux_vcr::<*field*>(R,L) outputs vcr_R
pub fn aux_vcr<F: FftField>(
    main: &MatrixEncoding<F>,
    sec: &MatrixEncoding<F>,
) -> DMatrix<DPolynomial<F>> {
    let row_main = &main.row_evals.evals;
    let col_main = &main.col_evals.evals;
    let val_main = &main.val_evals.evals;

    let row_sec = &sec.row_evals.evals;
    let col_sec = &sec.col_evals.evals;
    let val_sec = &sec.val_evals.evals;

    let r_c_v_l: Vec<F> = val_main
        .iter()
        .zip(row_main.iter())
        .zip(col_main.iter())
        .map(|((&val, &row), &col)| val * row * col)
        .collect();
    let r_c_v_l_poly =
        Evaluations::from_vec_and_domain(r_c_v_l.clone(), main.val_evals.domain()).interpolate();

    let r_c_v_l_r: Vec<F> = r_c_v_l
        .iter()
        .zip(row_sec.iter())
        .map(|(&rcvl, &r)| rcvl * r)
        .collect();
    let r_c_v_l_r_poly =
        Evaluations::from_vec_and_domain(r_c_v_l_r.clone(), main.val_evals.domain()).interpolate();

    let r_c_v_l_c: Vec<F> = r_c_v_l
        .iter()
        .zip(col_sec.iter())
        .map(|(&rcvl, &c)| rcvl * c)
        .collect();
    let r_c_v_l_c_poly =
        Evaluations::from_vec_and_domain(r_c_v_l_c, main.val_evals.domain()).interpolate();

    let r_c_v_l_r_c: Vec<F> = r_c_v_l_r
        .iter()
        .zip(col_sec.iter())
        .map(|(&rcvlr, &c)| rcvlr * c)
        .collect();
    let r_c_v_l_r_c_poly =
        Evaluations::from_vec_and_domain(r_c_v_l_r_c, main.val_evals.domain()).interpolate();

    vec![
        vec![r_c_v_l_r_c_poly, -r_c_v_l_r_poly],
        vec![-r_c_v_l_c_poly, r_c_v_l_poly],
    ]
}
