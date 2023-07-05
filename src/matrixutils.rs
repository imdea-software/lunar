use ark_ec::msm::FixedBaseMSM;
use ark_ec::msm::VariableBaseMSM;
use ark_ec::{prepare_g1, prepare_g2, AffineCurve};
use ark_ec::{PairingEngine, ProjectiveCurve};
use ark_ff::{FftField, Field, Zero};
use ark_poly::polynomial::Polynomial;
use ark_poly::{
    domain::EvaluationDomain,
    evaluations::univariate::Evaluations,
    univariate::{DenseOrSparsePolynomial, DensePolynomial as DPolynomial},
    GeneralEvaluationDomain, UVPolynomial,
};
use ark_std::rand::Rng;
use ark_std::rand::RngCore;
use std::ops::Mul;

use ark_ff::PrimeField;

/// Represents a square dense matrix (as a vector of rows).
pub type DMatrix<T> = Vec<Vec<T>>;

/// Represents a square sparse matrix (as a vector of sparse rows (coordinate, element)).
pub type SMatrix<T> = Vec<Vec<(usize, T)>>;

/// Wrapper of both dense and sparse matrices (maybe it should be called DenseOrSparseMatrix)
#[derive(Clone, PartialEq, Debug)]
pub enum Matrix<F> {
    DenseMatrix(DMatrix<F>),
    SparseMatrix(SMatrix<F>),
}

impl<F: FftField> Matrix<F> {
    /// Returns the number of rows of *self* (square matrix)
    pub fn nrows(&self) -> usize {
        match self {
            Matrix::DenseMatrix(dense) => dense.len(),
            Matrix::SparseMatrix(sparse) => sparse.len(),
        }
    }

    /// Returns the element in row,col (needed for the sparse matrix,
    /// I couldn't overload the index operator)
    pub fn elem(&self, row: usize, col: usize) -> F {
        match self {
            Matrix::DenseMatrix(dense) => dense[row][col],
            Matrix::SparseMatrix(sparse) => {
                let mut row_iter = sparse[row].iter();
                while let Some((c, e)) = row_iter.next() {
                    if *c == col {
                        return *e;
                    };
                }
                F::zero()
            }
        }
    }

    /// Count the number of non-zero elements in the square matrix *self*
    pub fn count_non_zeros(&self) -> usize {
        match self {
            Matrix::DenseMatrix(dense) => dense.iter().fold(0, |acc, n| {
                acc + n.iter().filter(|&elem| *elem != F::zero()).count()
            }),
            // Done to make sure entry with zero value are not counted
            Matrix::SparseMatrix(sparse) => sparse
                .iter()
                .map(|a: &Vec<(usize, F)>| {
                    a.iter().filter(|(size, elem)| *elem != F::zero()).count()
                })
                .fold(0, |acc, x| acc + x),
        }
    }

    /// Encode a matrix as a MatrixEncoding struct (rows, cols and vals are polynomials)
    pub fn encode_matrix(&self) -> MatrixEncoding<F> {
        let m = self.count_non_zeros();
        let n = self.nrows();
        let domain_k = GeneralEvaluationDomain::<F>::new(m).unwrap();
        let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();
        // Creation of row_v, col_v and val_v vectors that store the nonzeros of M
        let mut row_v = Vec::<F>::with_capacity(m);
        let mut col_v = Vec::<F>::with_capacity(m);
        let mut val_v = Vec::<F>::with_capacity(m);
        let h_elems: Vec<F> = domain_h.elements().collect();

        let mut current_row = 0;

        match self {
            Matrix::DenseMatrix(dense) => {
                for row in dense {
                    let mut current_col = 0;
                    for elem in row {
                        if *elem != F::zero() {
                            row_v.push(h_elems[current_row]);
                            col_v.push(h_elems[current_col]);
                            val_v.push(*elem);
                        }
                        current_col += 1;
                    }
                    current_row += 1;
                }
            }
            Matrix::SparseMatrix(sparse) => {
                for row in sparse {
                    for &(col, elem) in row {
                        row_v.push(h_elems[current_row]);
                        col_v.push(h_elems[col]);
                        val_v.push(elem);
                    }
                    current_row += 1;
                }
            }
        }

        let row_evals = Evaluations::from_vec_and_domain(row_v, domain_k);
        let col_evals = Evaluations::from_vec_and_domain(col_v, domain_k);
        let val_evals = Evaluations::from_vec_and_domain(val_v, domain_k);
        MatrixEncoding {
            domain_h,
            domain_k,
            row_evals,
            col_evals,
            val_evals,
        }
    }
}

/// Represents the encoding of a sparse matrix by the polynomials row_M, col_M and val_M.
pub struct MatrixEncoding<F: FftField> {
    //    pub name: String, // maybe necessary
    pub domain_h: GeneralEvaluationDomain<F>,
    pub domain_k: GeneralEvaluationDomain<F>,
    pub row_evals: Evaluations<F, GeneralEvaluationDomain<F>>,
    pub col_evals: Evaluations<F, GeneralEvaluationDomain<F>>,
    pub val_evals: Evaluations<F, GeneralEvaluationDomain<F>>,
}

/// poly_const!(*field*, *number*) creates a polynomial constant
/// in the given field. E.g. poly_const!(bls12_381_fr, 1)
#[macro_export]
macro_rules! poly_const {
    ($F: ty, $a: expr) => {
        DPolynomial::<$F>::from_coefficients_slice(&[$a])
    };
}

/// x!(*field*) is an alias for the single variable polynomial x in the field *field*
#[macro_export]
macro_rules! x {
    ($F: ty) => {
        DPolynomial::<$F>::from_coefficients_slice(&[<$F>::zero(), <$F>::one()])
    };
}

/// Functions that maps a domain of size n into [n] canonically
pub fn phi<F: FftField>(elem: &F, domain: GeneralEvaluationDomain<F>) -> Option<usize> {
    let elems: Vec<F> = domain.elements().collect();
    elems.iter().position(|&x| x == *elem)
}

/// Shift a given polynomial p by n (result: x^n * p)
pub fn shift_by_n<F: Field>(n: usize, p: &DPolynomial<F>) -> DPolynomial<F> {
    assert!(n > 0);
    let mut shifted = vec![F::zero(); n];
    shifted.extend_from_slice(&p.coeffs);
    DPolynomial::from_coefficients_vec(shifted)
}

/// Generates a random square sparse matrix with *density* elements.
/// Panics if *density* >= the number of matrix elements. (The sparser, the more efficient)
pub fn smatrix_rand<F: Field, R: rand_core::RngCore>(
    dim: usize,
    density: usize,
    rng: &mut R,
) -> Matrix<F> {
    assert!(dim * dim >= density);
    let mut places = 0;
    let mut smatrix: Vec<Vec<(usize, F)>> = Vec::with_capacity(dim);
    for i in 0..dim {
        smatrix.push(Vec::new());
    }
    while places < density {
        let row: usize = rng.gen_range(0..dim);
        let col: usize = rng.gen_range(0..dim);
        let f = F::rand(rng);
        if smatrix[row].iter().all(|&x| x.0 != col) {
            smatrix[row].push((col, f));
            places += 1;
        }
    }
    Matrix::SparseMatrix(smatrix)
}

/// Creates a polynomial on the given field that vanishes on a list of elements of *field*
pub fn vanishing_on_set<F: FftField>(set: &[F]) -> DPolynomial<F> {
    let mut vanishing = DPolynomial::<F>::from_coefficients_slice(&[F::one()]);
    for point in set {
        vanishing = vanishing.mul(&DPolynomial::<F>::from_coefficients_slice(&[
            -*point,
            F::one(),
        ]));
    }
    vanishing
}

/// Generates the lagrange polynomial at tau for the domain *domain*.
pub fn lagrange_at_tau<F: FftField>(domain: GeneralEvaluationDomain<F>, tau: F) -> DPolynomial<F> {
    let d_size = domain.size();
    let mut evaluations_vec = vec![F::zero(); d_size];
    let tau_index = domain
        .elements()
        .into_iter()
        .position(|r| r == tau)
        .unwrap();
    evaluations_vec[tau_index] = F::one();
    let evaluations = Evaluations::from_vec_and_domain(evaluations_vec, domain);
    evaluations.interpolate()
}

/// Generates the lagrange polynomial at tau for the given subdomain.
pub fn lagrange_on_subdomain_at_tau<F: FftField>(subdomain: &[F], tau: F) -> DPolynomial<F> {
    let lagrange = subdomain.iter().filter(|&x| x != &tau).fold(
        DPolynomial::<F>::from_coefficients_slice(&[F::one()]),
        |acc, &x| {
            acc.mul(
                &(&DPolynomial::<F>::from_coefficients_slice(&[-x, F::one()])
                    / (&DPolynomial::<F>::from_coefficients_slice(&[tau - x]))),
            )
        },
    );
    lagrange
}

/// Transform a polynomial into another in a way they agree over a given set/domain
pub fn masking_polynomial<F: FftField, R: rand_core::RngCore>(
    p: &DPolynomial<F>,
    b: usize,
    set: &[F],
    rng: &mut R,
) -> DPolynomial<F> {
    let rho = DPolynomial::rand(b - 1, rng);
    p + &rho.mul(&vanishing_on_set(set))
}

/// Lagrange bivariate function in the variable X
pub fn lambda_x<F: FftField>(x: F, domain_h: GeneralEvaluationDomain<F>) -> DPolynomial<F> {
    let n = domain_h.size() as u32;
    // Z_H evaluado en x
    &(&poly_const!(F, domain_h.evaluate_vanishing_polynomial(x)).mul(&x!(F))
        - &(poly_const!(F, x).mul(&(domain_h.vanishing_polynomial().into()))))
        / &(&poly_const!(F, F::from(n) * x) - &(&poly_const!(F, F::from(n)) * &x!(F)))
}

/// V_M function defined in the matrix encodings (definition 9)
/// sum over K of valM (κ) · LH rowM (κ)(X) · LcolM (κ)(Y).
/// with the variable X fixed on *x_point*
pub fn v_M_Y<F: FftField>(encoding: &MatrixEncoding<F>, x_point: F) -> DPolynomial<F> {
    let MatrixEncoding {
        domain_h,
        domain_k,
        row_evals,
        col_evals,
        val_evals,
    } = encoding;

    let v_M = val_evals
        .evals
        .iter()
        .zip(row_evals.evals.iter())
        .zip(col_evals.evals.iter())
        .fold(
            DPolynomial::<F>::from_coefficients_slice(&[F::zero()]),
            |acc, ((&val, &row), &col)| {
                acc + poly_const!(F, val)
                    .mul(&poly_const!(
                        F,
                        lagrange_at_tau(*domain_h, row).evaluate(&x_point)
                    ))
                    .mul(&lagrange_at_tau(*domain_h, col))
            },
        );
    v_M
}

/// Polynomial p' used PHPlite: sum(val_L+alpha*val_R)Lrow*Lcol*L(X)
pub fn p_prime<F: FftField>(
    encodingL: &MatrixEncoding<F>,
    encodingR: &MatrixEncoding<F>,
    alpha: F,
    x_point: F,
    y_point: F,
) -> DPolynomial<F> {
    let MatrixEncoding {
        domain_h: h,
        domain_k,
        row_evals: row_evals_L,
        col_evals: col_evals_L,
        val_evals: val_evals_L,
    } = encodingL;

    let MatrixEncoding {
        domain_h: h,
        domain_k,
        row_evals: row_evals_R,
        col_evals: col_evals_R,
        val_evals: val_evals_R,
    } = encodingR;

    let domain_k = GeneralEvaluationDomain::<F>::new(row_evals_L.evals.len()).unwrap();

    let p_prime_poly = domain_k
        .elements()
        .zip(val_evals_L.evals.iter())
        .zip(row_evals_L.evals.iter())
        .zip(col_evals_L.evals.iter())
        .zip(val_evals_R.evals.iter())
        .zip(row_evals_R.evals.iter())
        .zip(col_evals_R.evals.iter())
        .fold(
            poly_const!(F, F::zero()),
            |acc, ((((((k, &val_L), &row_L), &col_L), &val_R), &row_R), &col_R)| {
                acc + DPolynomial::<F>::from_coefficients_slice(&[val_L
                    .mul(&lagrange_at_tau(*h, row_L).evaluate(&x_point))
                    .mul(&lagrange_at_tau(*h, col_L).evaluate(&y_point))
                    + alpha.mul(
                        val_R
                            .mul(&lagrange_at_tau(*h, row_R).evaluate(&x_point))
                            .mul(&lagrange_at_tau(*h, col_R).evaluate(&y_point)),
                    )])
                .mul(&lagrange_at_tau(domain_k, k))
            },
        );
    p_prime_poly
}

/// Given a PrimeField and an element alpha, and an integer t, returns
/// a vector with [1, alpha, alpha^2...alpha^t]
pub fn gen_pows<Fr: PrimeField>(alpha: Fr, t: usize) -> Vec<Fr> {
    let mut powers_of_alpha = Vec::<Fr>::with_capacity(t + 1);
    powers_of_alpha.push(Fr::one());
    let mut power = alpha;

    for _ in 0..t {
        powers_of_alpha.push(power);
        power *= &alpha;
    }
    powers_of_alpha
}

/// Fixed Multi-scalar Multiplication in the curve G.
pub fn fmsm<G: ProjectiveCurve>(g: &G, v: &[G::ScalarField], t: usize) -> Vec<G> {
    let window_size = FixedBaseMSM::get_mul_window_size(t + 1);
    let scalar_bits = G::ScalarField::size_in_bits();

    let g_table = FixedBaseMSM::get_window_table(scalar_bits, window_size, *g);
    // from: multi_scalar_mul following update in ark-ec
    let powers_of_g = FixedBaseMSM::multi_scalar_mul::<G>(scalar_bits, window_size, &g_table, &v);

    powers_of_g
}

/// General deterministic poly commitment function: evaluates a poly on a point whose powers are given in the group
pub fn eval_on_pnt_in_grp<G: ProjectiveCurve>(
    p: &DPolynomial<G::ScalarField>,
    pnt_pws_in_grp: &Vec<G>,
) -> G {
    let g_vector_affine = G::batch_normalization_into_affine(pnt_pws_in_grp);
    // from: multi_scalar_mul following update in ark-ec
    let comm = VariableBaseMSM::multi_scalar_mul(
        &g_vector_affine,
        &p.coeffs.iter().map(|a| a.into_repr()).collect::<Vec<_>>(),
    ); // from: multi_scalar_mul

    comm
}

// Function for computing (p(X) - p(a)) / (X - a)
pub fn kate_division<F: FftField>(p: &DPolynomial<F>, a: F, b: F) -> DPolynomial<F> {
    let poly_x_a = &DPolynomial::<F>::from_coefficients_slice(&[-a, F::one()]);

    let (q, r) = DenseOrSparsePolynomial::from(p - &poly_const!(F, b))
        .divide_with_q_and_r(&poly_x_a.into())
        .unwrap();
    assert!(r.is_zero());

    return q;
}
