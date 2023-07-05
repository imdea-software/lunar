use ark_bls12_381::Bls12_381;
use ark_bls12_381::Fr as bls12_381_fr;

use rand::thread_rng;
use rand::Rng;

use std::time::Instant;

use crate::comms::*;
use crate::r1cslite::example_simple_v2;
use lunar_lib::*;

const NUM_SETUP_REPETITIONS: usize = 10;
const NUM_PROVE_REPETITIONS: usize = 10;
const NUM_VERIFY_REPETITIONS: usize = 50;

macro_rules! lunarlite_setup_bench {
    ($bench_name:ident, $bench_field:ty, $bench_pairing_engine:ty, $bench_commit_scheme:ty, $bench_cs_pow:expr, $bench_param:expr, $bench_id:expr) => {

        let mut rng = &mut thread_rng();
        // let rng = &mut ark_std::test_rng();

        let cs_max_degree = 2_usize.pow($bench_cs_pow);

        let (n_max,n_example,m,l) = $bench_param;

	    let n_next_pow2 = ((3*n_example) as usize).next_power_of_two();


        let now_global = Instant::now();

        for _ in 0..NUM_SETUP_REPETITIONS {

            let exampler1cslite = example_simple_v2::<$bench_field, _>(n_example, n_next_pow2, l, rng);

            let _ = LunarLite::<
                $bench_field,
                $bench_pairing_engine,
                $bench_commit_scheme
                >::setup_relation_encoder(
                    n_max, n_example, m, l, cs_max_degree, &mut rng, $bench_id, exampler1cslite);

        }

        let elapsed_time_global = now_global.elapsed();
        println!("Running setup took in average {} ms.\n", elapsed_time_global.as_millis() / NUM_SETUP_REPETITIONS as u128);

    };
}

macro_rules! lunarlite_prove_bench {
    ($bench_name:ident, $bench_field:ty, $bench_pairing_engine:ty, $bench_commit_scheme:ty, $bench_cs_pow:expr, $bench_param:expr) => {

        let mut rng = &mut thread_rng();
        // let rng = &mut ark_std::test_rng();

        let cs_max_degree = 2_usize.pow($bench_cs_pow);

        let (n_max, n_example, m, l) = $bench_param;
        let n_next_pow2 = ((3 * n_example) as usize).next_power_of_two();

        let exampler1cslite = example_simple_v2::<$bench_field, _>(n_example, n_next_pow2, l, rng);

        let ((prover_PP, verifier_PP, cs_pp),
                     (mut php_prover_view, mut php_verifier_view, mut prover_Cview),
                     _) = LunarLite::<
                    $bench_field,
                    $bench_pairing_engine,
                    $bench_commit_scheme
                    >::setup_relation_encoder(
                        n_max, n_example, m, l, cs_max_degree, &mut rng, "", exampler1cslite);

        let now = Instant::now();

        for _ in 0..NUM_PROVE_REPETITIONS {
            let _ = LunarLite::<$bench_field, $bench_pairing_engine, $bench_commit_scheme>::prover(
                &prover_PP,
                &verifier_PP,
                &cs_pp,
                &mut php_prover_view,
                &mut php_verifier_view,
                &mut prover_Cview,
            );
        }

        let elapsed_time = now.elapsed();
        println!(
            "Running prover took {} ms.\n",
            elapsed_time.as_millis() / NUM_PROVE_REPETITIONS as u128
        );
    };
}

macro_rules! lunarlite_verify_bench {
    ($bench_name:ident, $bench_field:ty, $bench_pairing_engine:ty, $bench_commit_scheme:ty, $bench_cs_pow:expr, $bench_param:expr) => {
        
        let mut rng = thread_rng();
        // let mut rng = &mut ark_std::test_rng();

        let cs_max_degree = 2_usize.pow($bench_cs_pow);

        let (n_max, n_example, m, l) = $bench_param;

        let n_next_pow2 = ((3 * n_example) as usize).next_power_of_two();

        let exampler1cslite =
            example_simple_v2::<$bench_field, _>(n_example, n_next_pow2, l, &mut rng);

        let now = Instant::now();

        let ((prover_PP, verifier_PP, cs_pp),
                     (mut php_prover_view, mut php_verifier_view, mut prover_Cview),
                     vk_d) = LunarLite::<
                    $bench_field,
                    $bench_pairing_engine,
                    $bench_commit_scheme
                    >::setup_relation_encoder(
                        n_max, n_example, m, l, cs_max_degree, &mut rng, "", exampler1cslite);

        let elapsed_time = now.elapsed();
        println!("Single setup took {} ms.", elapsed_time.as_millis());
        let now = Instant::now();

        let proofs =
            LunarLite::<$bench_field, $bench_pairing_engine, $bench_commit_scheme>::prover(
                &prover_PP,
                &verifier_PP,
                &cs_pp,
                &mut php_prover_view,
                &mut php_verifier_view,
                &mut prover_Cview,
            )
            .unwrap();

        let elapsed_time = now.elapsed();
        println!("Single Prover setup took {} ms.", elapsed_time.as_millis());
        let now = Instant::now();

        for _ in 0..NUM_VERIFY_REPETITIONS {
            let _ =
                LunarLite::<$bench_field, $bench_pairing_engine, $bench_commit_scheme>::verifier(
                    &verifier_PP,
                    &cs_pp,
                    &mut php_verifier_view,
                    &prover_Cview,
                    proofs.clone(),
                    vk_d.clone(),
                )
                .unwrap();
        }

        let elapsed_time = now.elapsed();
        println!(
            "Running verifier took {} ms.\n",
            elapsed_time.as_millis() / NUM_VERIFY_REPETITIONS as u128
        );
    };
}

macro_rules! lunar_setup_bench {
    ($bench_name:ident, $bench_field:ty, $bench_pairing_engine:ty, $bench_commit_scheme:ty, $bench_cs_pow:expr, $bench_param:expr) => {
        let mut rng = &mut thread_rng();
        // let rng = &mut ark_std::test_rng();

        let cs_max_degree = 2_usize.pow($bench_cs_pow);

        let (n_max, n_example, m, l) = $bench_param;
        
        let n_next_pow2 = ((3 * n_example) as usize).next_power_of_two();
        
        let now = Instant::now();
        for _ in 0..NUM_SETUP_REPETITIONS {
            let exampler1cslite =
                example_simple_v2::<$bench_field, _>(n_example, n_next_pow2, l, rng);

            let _ = Lunar::<
                        $bench_field,
                        $bench_pairing_engine,
                        $bench_commit_scheme
                        >::setup_relation_encoder(
                            n_max, n_example, m, l, exampler1cslite, cs_max_degree, &mut rng);
        }

        let elapsed_time = now.elapsed();
        println!(
            "Running setup took {} ms.\n",
            elapsed_time.as_millis() / NUM_SETUP_REPETITIONS as u128
        );
    };
}

macro_rules! lunar_prove_bench {
    ($bench_name:ident, $bench_field:ty, $bench_pairing_engine:ty, $bench_commit_scheme:ty, $bench_cs_pow:expr, $bench_param:expr) => {
        let mut rng = &mut thread_rng();
        // let rng = &mut ark_std::test_rng();
        let cs_max_degree = 2_usize.pow($bench_cs_pow);

        let (n_max, n_example, m, l) = $bench_param;
        let n_next_pow2 = ((3 * n_example) as usize).next_power_of_two();

        let exampler1cslite = example_simple_v2::<$bench_field, _>(n_example, n_next_pow2, l, rng);

        let ((prover_PP, verifier_PP, cs_pp),
                     (mut php_prover_view, mut php_verifier_view, mut prover_Cview)) = Lunar::<
                    $bench_field,
                    $bench_pairing_engine,
                    $bench_commit_scheme
                    >::setup_relation_encoder(
                        n_max, n_example, m, l, exampler1cslite, cs_max_degree, &mut rng);

        let now = Instant::now();

        for _ in 0..NUM_PROVE_REPETITIONS {
            let proofs = Lunar::<$bench_field, $bench_pairing_engine, $bench_commit_scheme>::prover(
                &prover_PP,
                &verifier_PP,
                &cs_pp,
                &mut php_prover_view,
                &mut php_verifier_view,
                &mut prover_Cview,
            );
        }

        let elapsed_time = now.elapsed();
        println!(
            "Running prover took {} ms.\n",
            elapsed_time.as_millis() / NUM_PROVE_REPETITIONS as u128
        );
    };
}

macro_rules! lunar_verify_bench {
    ($bench_name:ident, $bench_field:ty, $bench_pairing_engine:ty, $bench_commit_scheme:ty, $bench_cs_pow:expr, $bench_param:expr) => {
        let mut rng = &mut thread_rng();
        // let rng = &mut ark_std::test_rng();

        let cs_max_degree = 2_usize.pow($bench_cs_pow);

        let (n_max, n_example, m, l) = $bench_param;

        let n_next_pow2 = ((3 * n_example) as usize).next_power_of_two();

        let exampler1cslite = example_simple_v2::<$bench_field, _>(n_example, n_next_pow2, l, rng);


        let ((prover_PP, verifier_PP, cs_pp),
                     (mut php_prover_view, mut php_verifier_view, mut prover_Cview)
                     ) = Lunar::<
                        $bench_field,
                        $bench_pairing_engine,
                        $bench_commit_scheme
                        >::setup_relation_encoder(
                            n_max, n_example, m, l, exampler1cslite, cs_max_degree, &mut rng);


        let proofs = Lunar::<$bench_field, $bench_pairing_engine, $bench_commit_scheme>::prover(
            &prover_PP,
            &verifier_PP,
            &cs_pp,
            &mut php_prover_view,
            &mut php_verifier_view,
            &mut prover_Cview,
        )
        .unwrap();

        let now = Instant::now();
        for _ in 0..NUM_VERIFY_REPETITIONS {
            let _ = Lunar::<$bench_field, $bench_pairing_engine, $bench_commit_scheme>::verifier(
                &verifier_PP,
                &cs_pp,
                &mut php_verifier_view,
                &prover_Cview,
                proofs.clone(),
            )
            .unwrap();
        }

        let elapsed_time = now.elapsed();
        println!(
            "Running verifier took {} ms.\n",
            elapsed_time.as_millis() / NUM_VERIFY_REPETITIONS as u128
        );
    };
}

fn bench_setup_lunarlite() {
    println!("\nBENCHMARK FOR THE SETUP START\n");


    let power = 20;
    let cs_max_degree = 2_usize.pow(power);
    let n_max = cs_max_degree;
    let n_example = 150;
    let m = 3 * n_example;
    let l = 30;

    println!("PARAMETER SETUP: power = {:?}, cs_max_degree = {:?}, nmax = {:?}, n_example = {:?}, m = {:?}, l = {:?}", power, cs_max_degree, n_max,n_example,m,l);
    lunarlite_setup_bench!(
        bls,
        bls12_381_fr,
        Bls12_381,
        CS2,
        power,
        (n_max, n_example, m, l),
        "_id"
    );

    println!("BENCHMARK FOR THE SETUP DONE\n\n\n");
}

fn bench_prove_lunarlite() {
    println!("\n\n\nBENCHMARK FOR THE PROVER START");

    let power = 20;
    let cs_max_degree = 2_usize.pow(power);
    let n_max = cs_max_degree;
    let n_example = 150;
    let m = 3 * n_example;
    let l = 30;

    println!("PARAMETER SETUP: power = {:?}, cs_max_degree = {:?}, nmax = {:?}, n_example = {:?}, m = {:?}, l = {:?}", power, cs_max_degree, n_max,n_example,m,l);
    lunarlite_prove_bench!(
        bls,
        bls12_381_fr,
        Bls12_381,
        CS2,
        power,
        (n_max, n_example, m, l)
    );

    println!("BENCHMARK FOR THE PROVER DONE\n\n\n");
}

fn bench_verify_lunarlite() {
    println!("\n\n\nBENCHMARK FOR THE VERIFIER START");

    let power = 20;
    let cs_max_degree = 2_usize.pow(power);
    let n_max = cs_max_degree;
    let n_example = 150;
    let m = 3 * n_example;
    let l = 30;

    println!("PARAMETER SETUP: power = {:?}, cs_max_degree = {:?}, nmax = {:?}, n_example = {:?}, m = {:?}, l = {:?}", 
                10, cs_max_degree, n_max,n_example,m,l);
    lunarlite_verify_bench!(
        bls,
        bls12_381_fr,
        Bls12_381,
        CS2,
        10,
        (n_max, n_example, m, l)
    );

    println!("BENCHMARK FOR THE VERIFIER DONE\n\n\n");
}

fn bench_setup_lunar() {
    println!("\n\n\nBENCHMARK FOR THE LUNAR SETUP START\n");

    let power = 20;
    let cs_max_degree = 2_usize.pow(power);
    let n_max = cs_max_degree;
    let n_example = 150;
    let m = 3 * n_example;
    let l = 30;


    println!("PARAMETER SETUP: power = {:?}, cs_max_degree = {:?}, nmax = {:?}, n_example = {:?}, m = {:?}, l = {:?}", power, cs_max_degree, n_max,n_example,m,l);
    lunar_setup_bench!(
        bls,
        bls12_381_fr,
        Bls12_381,
        CS1,
        power,
        (n_max, n_example, m, l)
    );

    println!("BENCHMARK FOR THE LUNAR SETUP DONE\n\n\n");
}

fn bench_prove_lunar() {
    println!("\n\n\nBENCHMARK FOR THE LUNAR PROVER START");

    let power = 20;
    let cs_max_degree = 2_usize.pow(power);
    let n_max = cs_max_degree;
    let n_example = 150;
    let m = 3 * n_example;
    let l = 30;

    println!("PARAMETER SETUP: power = {:?}, cs_max_degree = {:?}, nmax = {:?}, n_example = {:?}, m = {:?}, l = {:?}", power, cs_max_degree, n_max,n_example,m,l);
    lunar_prove_bench!(
        bls,
        bls12_381_fr,
        Bls12_381,
        CS1,
        power,
        (n_max, n_example, m, l)
    );

    println!("BENCHMARK FOR THE LUNAR PROVER DONE\n\n\n");
}

fn bench_verify_lunar() {
    println!("\n\n\nBENCHMARK FOR THE LUNAR VERIFIER START");

    let power = 20;
    let cs_max_degree = 2_usize.pow(power);
    let n_max = cs_max_degree;
    let n_example = 150;
    let m = 3 * n_example;
    let l = 30;

    println!("PARAMETER SETUP: power = {:?}, cs_max_degree = {:?}, nmax = {:?}, n_example = {:?}, m = {:?}, l = {:?}", power, cs_max_degree, n_max,n_example,m,l);
    lunar_verify_bench!(
        bls,
        bls12_381_fr,
        Bls12_381,
        CS1,
        power,
        (n_max, n_example, m, l)
    );
    
    println!("BENCHMARK FOR THE LUNAR VERIFIER DONE\n\n\n");
}

fn main() {
    
    bench_setup_lunarlite();
    bench_prove_lunarlite();
    bench_verify_lunarlite();

    bench_setup_lunar();
    bench_prove_lunar();
    bench_verify_lunar();

}
