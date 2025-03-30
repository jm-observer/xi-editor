use lapce_xi_rope::Interval;
use lapce_xi_rope::spans::SpansBuilder;

fn main() {
    let mut span = SpansBuilder::new(1000);
    for start in 1..75 {
        let start = start * 10;
        span.add_span(Interval::new(start, start + 5), start);
    }
    let span = span.build();
    span.print_tree(0);

    for i in span.iter_chunks(300..400) {
        println!("{:?} - {}", i.0, i.1)
    }
}
