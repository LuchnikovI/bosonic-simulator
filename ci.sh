#!/usr/bin/env bash

ci_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

builder="${ci_dir}/builder"


echo "INFO: Building an image and compiling code within"

cat << EOF > "${builder}.def"
Bootstrap: docker
From: rust:1.71-alpine3.18

%files
    "${ci_dir}/.cargo" /.cargo
    "${ci_dir}/src" /src
    "${ci_dir}/Cargo.toml" /Cargo.toml
    "${ci_dir}/Cargo.lock" /Cargo.lock

%post
    apk add musl-dev && rustup target add x86_64-unknown-linux-musl
    cargo build --release
    cargo test --no-run 2> /compilation_log
    mv "/target/x86_64-unknown-linux-musl/debug/deps/\$(grep -o "bosonic_processor-[^)]*" /compilation_log)" "/test"
    mv "/target/x86_64-unknown-linux-musl/release/bosonic_processor" "/bosonic_processor"
EOF

if singularity build -F "${builder}.sif" "${builder}.def"; then
    echo "INFO: Image has been built"
    rm ${builder}.def
else
    echo "ERROR: Unable to build an image"
    rm ${builder}.def
    exit 1
fi

echo "INFO: Extracing artefacts from the container"


singularity exec ${builder}.sif cp /test "${ci_dir}/test" && \
singularity exec ${builder}.sif cp /bosonic_processor "${ci_dir}/tasks/bosonic_processor"

if [[ $? -ne 0 ]]; then
    echo "ERROR: Unable to extract artefacts"
    rm ${builder}.sif
    exit 1
else
    echo "INFO: Artefacts have been extracted"
    rm ${builder}.sif
fi

if ${ci_dir}/test; then
    echo "INFO: Tests OK"
    rm "${ci_dir}/test"
else
    echo "ERROR: Tests failed"
    rm "${ci_dir}/test"
    rm "${ci_dir}/bosonic_processor"
    exit 1
fi
