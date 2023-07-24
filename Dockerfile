FROM  rust:1.71-alpine3.18 AS builder
WORKDIR /bosonic_emulator
COPY . .
RUN apk add musl-dev && \
    rustup target add x86_64-unknown-linux-musl && \
    cargo build --release && \
    cargo test --no-run 2> ./compilation_log
RUN mv "./target/x86_64-unknown-linux-musl/debug/deps/$(grep -o "bosonic_processor-[^)]*" ./compilation_log)" ./test && \
    mv ./target/x86_64-unknown-linux-musl/release/bosonic_processor ./bosonic_processor


FROM scratch
COPY --from=builder /bosonic_emulator/bosonic_processor \
    /bosonic_processor
COPY --from=builder /bosonic_emulator/test /test

CMD [ "/test" ]
