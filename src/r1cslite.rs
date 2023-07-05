#![allow(non_snake_case)]

use ark_ff::{FftField, Field, Zero};
use ark_poly::polynomial::Polynomial;
use ark_poly::{
    domain::EvaluationDomain,
    evaluations::univariate::Evaluations,
    univariate::{DenseOrSparsePolynomial, DensePolynomial as DPolynomial},
    GeneralEvaluationDomain, UVPolynomial,
};

// use serde::{Serialize, Deserialize};

use ark_std::rand::RngCore;
use rand::Rng;
use std::time::Instant;

//use rand::Rng;
//use rand_core::RngCore;
use rand::thread_rng;

use crate::matrixutils::*;
use crate::{poly_const, x};
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct ExampleR1CSLite<F: FftField> {
    pub matrices: (Matrix<F>, Matrix<F>),
    pub polynomials: (DPolynomial<F>, DPolynomial<F>),
    pub vector: Vec<F>,
}

/// Inputs a R1CSLite instance and outputs a longR1CSlite instance
/// Concretely the polynomials a', b' and the witness in vector form.
fn r1cslite_to_longr1cslite<F: FftField>(
    n: usize,
    m: usize,
    l: usize,
    x: &Vec<F>,
    w: Vec<F>,
    L: &Matrix<F>,
    R: &Matrix<F>,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let mut c_hat = vec![F::one()];
    c_hat.extend(x);
    c_hat.extend(&w);

    // calculo de a_prime, b_prime
    let mut a_prime = Vec::<F>::new();
    let mut b_prime = Vec::<F>::new();

    for nrow in l..n {
        match L {
            Matrix::DenseMatrix(denseL) => {
                a_prime.push(
                    denseL[nrow]
                        .iter()
                        .zip(c_hat.iter())
                        .fold(F::zero(), |acc, (&l, &c)| acc - l * c),
                );
            }
            Matrix::SparseMatrix(sparseL) => {
                a_prime.push(
                    sparseL[nrow]
                        .iter()
                        .fold(F::zero(), |acc, (col, elem)| acc - *elem * c_hat[*col]),
                );
            }
        };

        match R {
            Matrix::DenseMatrix(denseR) => {
                b_prime.push(
                    denseR[nrow]
                        .iter()
                        .zip(c_hat.iter())
                        .fold(F::zero(), |acc, (&r, &c)| acc - r * c),
                );
            }
            Matrix::SparseMatrix(sparseR) => {
                b_prime.push(
                    sparseR[nrow]
                        .iter()
                        .fold(F::zero(), |acc, (col, elem)| acc - *elem * c_hat[*col]),
                );
            }
        };
    }

    assert_eq!(
        a_prime
            .iter()
            .zip(b_prime.iter())
            .map(|(&a, &b)| a * b)
            .collect::<Vec<F>>(),
        w
    );

    (a_prime, b_prime, w)
}

/// Tranforms a longR1CSlite instance into a PolyR1CSlite instance
fn longr1cslite_to_polyr1cslite<F: FftField>(
    n: usize,
    m: usize,
    l: usize,
    x: Vec<F>,
    a_prime: Vec<F>,
    b_prime: Vec<F>,
    //L: &DMatrix<F>,
    //R: &DMatrix<F>,
) -> (DPolynomial<F>, DPolynomial<F>) {
    // a = (1, x, a') len(a) = n
    let mut a = vec![F::one()];
    a.extend(x.clone());
    a.extend(a_prime);

    // b = (1..1, b') len(b) = n
    let mut b = vec![F::one(); l];
    b.extend(b_prime);

    // x' = (1, x) len(x') = l
    let mut x_prime = vec![F::one()];
    x_prime.extend(x);

    let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();
    let poly_a = Evaluations::<F>::from_vec_and_domain(a.clone(), domain_h).interpolate();
    let poly_b = Evaluations::<F>::from_vec_and_domain(b.clone(), domain_h).interpolate();
    let poly_one = poly_const!(F, F::one());

    let poly_inter = Evaluations::<F>::from_vec_and_domain(x_prime.clone(), domain_h).interpolate();

    let LL = &domain_h.elements().collect::<Vec<F>>()[0..l];
    let poly_num_a =
        &poly_a - &Evaluations::<F>::from_vec_and_domain(x_prime.clone(), domain_h).interpolate();
    let poly_num_b = &poly_b - &poly_one;

    let poly_dem = vanishing_on_set(LL);
    let (poly_a_prime, remainder_a) =
        DenseOrSparsePolynomial::divide_with_q_and_r(&(&poly_num_a).into(), &(&poly_dem).into())
            .unwrap();
    assert!(remainder_a.is_zero());
    let (poly_b_prime, remainder_b) =
        DenseOrSparsePolynomial::divide_with_q_and_r(&(&poly_num_b).into(), &(&poly_dem).into())
            .unwrap();
    assert!(remainder_b.is_zero());

    // Added to check equality
    let a_hat = poly_a_prime.mul(&poly_dem)
        + LL.iter().fold(poly_const!(F, F::zero()), |acc, &tau| {
            acc + &poly_const!(F, x_prime[phi(&tau, domain_h).unwrap()])
                * &lagrange_at_tau(domain_h, tau)
        });

    (poly_a_prime, poly_b_prime)
}

/// Tranforms a R1CSLite instance into a PolyR1CSlite one.
pub fn r1cslite_to_polyr1cslite<F: FftField>(
    nmax: usize,
    m: usize,
    l: usize,
    x: Vec<F>,
    w: Vec<F>,
    L: &Matrix<F>,
    R: &Matrix<F>,
) -> (DPolynomial<F>, DPolynomial<F>) {
    let now = Instant::now();

    let (a_prime, b_prime, c_prime) = r1cslite_to_longr1cslite(nmax, m, l, &x, w, &L, &R);

    let elapsed_time = now.elapsed();
    // println!("Example r1cslite_to_longr1cslite took {} microsec.", elapsed_time.as_micros());
    let now = Instant::now();

    let temp = longr1cslite_to_polyr1cslite(nmax, m, l, x, a_prime, b_prime); //, &L, &R)

    let elapsed_time = now.elapsed();
    // println!("Example longr1cslite_to_polyr1cslite took {} microsec.", elapsed_time.as_micros());

    temp
}

/// Verify the equations (17-18)
/// a(eta) + sum(L_eta,eta_prime*a(eta_prime)*b(eta_prime)) = 0
/// b(eta) + sum(R_eta,eta_prime*a(eta_prime)*b(eta_prime)) = 0
/// for all eta in domain_h
pub fn check_scalar<F: FftField>(
    poly1: &DPolynomial<F>,
    poly2: &DPolynomial<F>,
    matrix: &Matrix<F>,
    domain: GeneralEvaluationDomain<F>,
) -> bool {
    domain.elements().all(|eta| {
        (poly1.evaluate(&eta)
            + domain.elements().fold(F::zero(), |acc, eta_prime| {
                acc + matrix.elem(phi(&eta, domain).unwrap(), phi(&eta_prime, domain).unwrap())
                    * poly1.evaluate(&eta_prime)
                    * poly2.evaluate(&eta_prime)
            }))
        .is_zero()
    })
}

/// Polynomial version of the equations 17-18 check
pub fn check_poly<F: FftField>(
    poly1: &DPolynomial<F>,
    poly2: &DPolynomial<F>,
    matrix: &Matrix<F>,
    domain: GeneralEvaluationDomain<F>,
) -> bool {
    let mut v = vec![];
    for eta in domain.elements() {
        v.push(domain.elements().fold(F::zero(), |acc, eta_prime| {
            acc + matrix.elem(phi(&eta, domain).unwrap(), phi(&eta_prime, domain).unwrap())
                * poly1.evaluate(&eta_prime)
                * poly2.evaluate(&eta_prime)
        }))
    }
    (poly1 + &Evaluations::from_vec_and_domain(v, domain).interpolate()).is_zero()
}

pub fn check_poly_prime<F: FftField>(
    a_prime_poly: &DPolynomial<F>,
    b_prime_poly: &DPolynomial<F>,
    l: usize,
    matrixL: &Matrix<F>,
    matrixR: &Matrix<F>,
    domain: GeneralEvaluationDomain<F>,
    x: &Vec<F>,
) -> (bool, bool) {
    let mut x_prime = vec![F::one()];
    x_prime.extend(x);
    let LL = &domain.elements().collect::<Vec<F>>()[0..l];
    let vanishingLL = vanishing_on_set(LL);

    let poly_inter = Evaluations::<F>::from_vec_and_domain(x_prime.clone(), domain).interpolate();
    let a_poly = Evaluations::from_vec_and_domain(x_prime, domain).interpolate()
        + a_prime_poly * &vanishingLL;

    let b_poly = poly_const!(F, F::one()) + b_prime_poly * &vanishingLL;
    let mut v1 = vec![];
    let mut v2 = vec![];
    for eta in domain.elements() {
        v1.push(domain.elements().fold(F::zero(), |acc, eta_prime| {
            let sum = matrixL.elem(phi(&eta, domain).unwrap(), phi(&eta_prime, domain).unwrap())
                * a_poly.evaluate(&eta_prime)
                * b_poly.evaluate(&eta_prime);
            acc + sum
        }))
    }
    for eta in domain.elements() {
        v2.push(domain.elements().fold(F::zero(), |acc, eta_prime| {
            acc + matrixR.elem(phi(&eta, domain).unwrap(), phi(&eta_prime, domain).unwrap())
                * a_poly.evaluate(&eta_prime)
                * b_poly.evaluate(&eta_prime)
        }))
    }

    (
        (a_poly + Evaluations::from_vec_and_domain(v1, domain).interpolate()).is_zero(),
        (b_poly + Evaluations::from_vec_and_domain(v2, domain).interpolate()).is_zero(),
    )
}

/// Simple instance for the R1CSLite problem. Where the
/// L matrix is of size nxn with -1 in the diagonal of the first l rows
/// and the identity matrix in the n-l last rows
pub fn simple_matrixL<F: Field>(n: usize, l: usize) -> Matrix<F> {
    let mut L = Vec::<Vec<(usize, F)>>::with_capacity(n);
    for i in 0..l {
        L.push(vec![(i, -F::one())]);
    }
    for i in l..n {
        L.push(vec![(i, F::one())]);
    }
    Matrix::SparseMatrix(L)
}

/// Simple instance for the R1CSLite problem. Where the
/// R matrix is of size nxn with -1 in the first position in the first l rows
/// and the identity matrix in the n-l last rows
pub fn simple_matrixR<F: Field>(n: usize, l: usize) -> Matrix<F> {
    let mut R = Vec::<Vec<(usize, F)>>::with_capacity(n);
    for i in 0..l {
        R.push(vec![(0, -F::one())]);
    }
    for i in l..n {
        R.push(vec![(i, F::one())]);
    }
    Matrix::SparseMatrix(R)
}

/// Simple instance for the R1CSLite problem.
/// L = (In|On|On)   with In = Identity matrix of size n
///     (   Un   )        On = Zero matrix of size n
///     (In|On|On)        Un = Matrix of size n with ones at the first position
/// nmax: next power of two of 3*n
pub fn simple_matrixL_v2<F: Field>(n: usize, nmax: usize, l: usize) -> Matrix<F> {
    let mut L = Vec::<Vec<(usize, F)>>::with_capacity(nmax);
    for i in 0..l {
        L.push(vec![(i, -F::one())]);
    }
    for i in l..n {
        L.push(vec![(i, F::one())]);
    }
    for i in 0..n {
        L.push(vec![(0, F::one())]);
    }
    for i in 0..n {
        L.push(vec![(i, F::one())]);
    }
    for i in (3 * n)..nmax {
        L.push(vec![(0, F::zero())]);
    }
    Matrix::SparseMatrix(L)
}

/// Simple instance for the R1CSLite problem.
/// R = (   Un   )   with In = Identity matrix of size n
///     (On|In|On)        On = Zero matrix of size n
///     (On|In|On)        Un = Matrix of size n with ones at the first position
/// nmax: next power of two of 3*n
pub fn simple_matrixR_v2<F: Field>(n: usize, nmax: usize, l: usize) -> Matrix<F> {
    let mut R = Vec::<Vec<(usize, F)>>::with_capacity(nmax);
    for i in 0..l {
        R.push(vec![(0, -F::one())]);
    }
    for i in l..n {
        R.push(vec![(0, F::one())]);
    }
    for i in 0..n {
        R.push(vec![(n + i, F::one())]);
    }
    for i in 0..n {
        R.push(vec![(n + i, F::one())]);
    }
    for i in (3 * n)..nmax {
        R.push(vec![(0, F::zero())]);
    }
    Matrix::SparseMatrix(R)
}

/// Complete instance of the simple R1CSLite problem, with L and R as described
/// before, x = [1...1] of size l-1 and w = [1..1] of size n-l. This function returns
/// the polyR1CSlite transformation of the instance described.
pub fn example_simple<F: FftField>(n: usize, l: usize) -> (DPolynomial<F>, DPolynomial<F>) {
    let L = simple_matrixL::<F>(n, l);
    let R = simple_matrixR::<F>(n, l);
    let x = vec![F::one(); l - 1];
    let w = vec![F::one(); n - l];
    let m = n;
    let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();
    let (a_prime_poly, b_prime_poly) = r1cslite_to_polyr1cslite(n, m, l, x.clone(), w, &L, &R);
    (a_prime_poly, b_prime_poly)
}

/// L = (In|On|On) R = (   Un   )
///     (   Un   )     (On|In|On)
///     (In|On|On)     (On|In|On)
///
/// with In = Identity matrix of size n
///      On = Zero matrix of size n
///      Un = Matrix of size n with ones at the first position
///
pub fn example_simple_v2<F: FftField, R: rand_core::RngCore>(
    n: usize,
    n_next_pow2: usize,
    l: usize,
    rng: &mut R,
) -> ExampleR1CSLite<F> //((DPolynomial<F>, DPolynomial<F>), Vec<F>)
{
    // println!("\nvalue of n = {:?}", n);
    // println!("value of l = {:?}", l);
    // println!("value of next power of two of 3n = {:?}\n", n_next_pow2);

    assert!(l <= n, "l should always be lesser or equal to n");

    // let mut rng = thread_rng();

    let now = Instant::now();

    let L = simple_matrixL_v2::<F>(n, n_next_pow2, l);
    let R = simple_matrixR_v2::<F>(n, n_next_pow2, l);

    let elapsed_time = now.elapsed();
    // println!("Example matrices generation took {} microsec.", elapsed_time.as_micros());
    let now = Instant::now();

    let mut x = Vec::<F>::with_capacity(l - 1);
    for i in 0..l - 1 {
        // Can be random value
        x.push(F::rand(rng));
    }
    let mut x_prime = Vec::<F>::new();
    x_prime.push(F::one());
    x_prime.extend(x.clone());

    let mut w = Vec::<F>::with_capacity(n_next_pow2 - l);
    for i in l..n {
        // Can be random value
        // w.push(F::from(2*i as u32));
        w.push(F::rand(rng));
    }
    for i in 0..n {
        // w.push(F::from(2*i as u32));
        w.push(F::rand(rng));
    }
    for i in 0..l {
        w.push(x_prime[i] * w[n - l + i]);
    }
    for i in 0..(n - l) {
        w.push(w[i] * w[n + i]);
    }
    for i in (3 * n)..(n_next_pow2) {
        w.push(F::zero());
    }

    //println!("1 = {:?}", F::from(1 as u32));;

    let m = 3 * n;

    let domain_h = GeneralEvaluationDomain::<F>::new(n_next_pow2).unwrap();

    let ((a_prime_poly, b_prime_poly), x_vec) = (
        r1cslite_to_polyr1cslite(n_next_pow2, m, l, x.clone(), w, &L, &R),
        x.clone(),
    );

    let elapsed_time = now.elapsed();
    // println!("Example r1cslite_to_polyr1cslite took {} microsec.", elapsed_time.as_micros());
    // println!("\nExample generation took {} microsec.", elapsed_time.as_micros());
    let now = Instant::now();

    ExampleR1CSLite {
        matrices: (L, R),
        polynomials: (a_prime_poly, b_prime_poly),
        vector: x_vec,
    }
}
