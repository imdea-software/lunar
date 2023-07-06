# Lunar

`lunar` is a Rust library that implements (some of) the Lunar zkSNARKs and corresponding building blocks proposed in [1].

**WARNING:** This is an academic prototype, and in particular has not received careful code review. This implementation is NOT ready for production use.

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

To run this project you need to [install the last version of Rust](https://www.rust-lang.org/tools/install). It's recommended to run the project via the Rust package manager, `cargo`, installed by default with the `rustup` toolchain.

This project uses the [arkworks](https://github.com/arkworks-rs/) ecosystem, mainly for finite fields (`ark-ff`), polynomial manipulation (`ark-poly`), elliptic curves and pairings (`ark-ec`), grouped in the repository [algebra].

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

## License
This code is licensed under either of the following licenses, at your discretion.

- [Apache License Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

Unless you explicitly state otherwise, any contribution that you submit to this library shall be dual licensed as above (as defined in the Apache v2 License), without any additional terms or conditions.

## Reference paper and contributors

[CFFQR21] Matteo Campanelli, Antonio Faonio, Dario Fiore, Anaïs Querol, and Hadrián Rodríguez. Lunar: a Toolbox for More Efficient Universal and Updatable zkSNARKs and Commit-and-Prove Extensions. ASIACRYPT 2021. [https://eprint.iacr.org/2020/1069](https://eprint.iacr.org/2020/1069)

This library has been developed with the contributions of: Matteo Campanelli, Hadrián Rodríguez, and Damien Robissout.

## Acknowledgements
This work has received funding by: the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program under project PICOCRYPT (grant agreement No. 101001283), by the Spanish Government under projects SCUM (ref. RTI2018-102043-B-I00) and CRYPTOEPIC (ref. EUR2019-103816), by the Madrid regional government under project BLOQUES (ref. S2018/TCS-4339), and by research grants from Protocol Labs, and from the Tezos foundation and Nomadic Labs. Additionally, the project that gave rise to these results received the support of a fellowship from “la Caixa” Foundation (ID 100010434) to Anaïs Querol (ref. LCF/BQ/ES18/11670018).

[rust book]: https://doc.rust-lang.org/book/
[algebra]: https://github.com/arkworks-rs/algebra "Rust arkworks-rs algebra"
[2]: https://www.rust-lang.org/tools/install "Rust installation"
