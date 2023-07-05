# Lunar: a Toolbox for More Efficient Universal and Updatable zkSNARKs and Commit-and-Prove Extensions


(Partial) Rust implementation of the zkSNARK compiler in [1]. (complete)

## Table of Contents
* [ Overview ](#overview)
* [ How to install ](#how-to-install)
* [ Usage ](#usage)
* [ Benchmarks ](#benchmarks)
* [ Tests ](#tests)
* [ To-do ](#to-do)

## Overview

Description of the different modules:
 - phplite2: Implements the prover and verifier for R1CSlite2 and R1CSlite2x;
 - comms: Implements the typed commitment scheme for CS1 and CS2;
 - evalpolyproofs: Implements the Commit-and-Prove gadgets for CS1 and CS2;
 - matrixutils: Utility function for matrix and polynomials;
 - php: Implements oracle structures for polynomials and commitments;
 - r1cslite: Implements the R1CSlite to polyR1CSlite conversion.

## How to install

To run this project you need to install the last version of Rust (see [2]). It's recommended to run the project via the Rust package manager, `cargo`, installed by default with the `rustup` toolchain.

This project uses the arkworks ecosystem [arkworks], mainly for finite fields (`ark-ff`), polynomial manipulation (`ark-poly`), elliptic curves and pairings (`ark-ec`), grouped in the repository [algebra].

## Usage

You can build this project with

`git clone https://gitlab.software.imdea.org/hadrian.rodriguez/lunar-implementation`

After logging in with your credentials,

```bash
cd lunar
cargo build --release
cargo run
```
You can build the documentation with 
`cargo doc` and navigate with `cargo doc --open`

## Benchmarks

The benchmarking is in the folder `lunar/benches`. Just run `cargo bench` to check the performance of our library.

## Tests
The tests are intended to check the validity of some of the functions involved.
To run the tests, just use `cargo test`.

## To-do:
 - The unwrap() function is not recommended because if an error prompts, it makes the program panic with no more information given. Thus it's recommended to transform all `unwrap()` into `expect("Some custom msg")`.



[rust book]: https://doc.rust-lang.org/book/
[algebra]: https://github.com/arkworks-rs/algebra "Rust arkworks-rs algebra"
[arkworks]: https://github.com/arkworks-rs/
[2]: https://www.rust-lang.org/tools/install "Rust installation"
[1]: https://eprint.iacr.org/2020/1069 "Lunar eprint"
