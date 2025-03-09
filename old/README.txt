The cargo.toml for the Serial_slow is the following:
Where i installed CUDA:
    C:\Users\eduar\AppData\Local\Temp\cuda

Where i installed arrayfire


[package]
name = "SymbolicRegression"
version = "0.1.0"
edition = "2024"

[dependencies]
ndarray = "0.16.1"    # For efficient data handling
rand = "0.8.5"        # For random number generation
rayon = "1.10.0"      # For parallel processing
statrs = "0.18.0"     # For statistical functions
