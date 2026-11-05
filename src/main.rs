use std::io;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let stdout = io::stdout();
    let stderr = io::stderr();
    let code = matrixlab::cli::run(&args, &mut stdout.lock(), &mut stderr.lock());
    std::process::exit(code);
}
