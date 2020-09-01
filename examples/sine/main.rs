use havana::{AverageAndErrorAccumulator, ContinuousGrid, Sample};

fn main() {
    let f = |x: f64| (x * std::f64::consts::PI).sin();
    let mut grid = ContinuousGrid::new(1, 10, 1000);

    let mut rng = rand::thread_rng();

    let mut integral = AverageAndErrorAccumulator::new();

    for _ in 1..20 {
        for _ in 0..10000 {
            let mut sample = Sample::ContinuousGrid(0., vec![]);

            grid.sample(&mut rng, &mut sample);
            if let Sample::ContinuousGrid(cont_weight, cs) = &sample {
                let res = f(cs[0]);
                grid.add_training_sample(&sample, res, false);
                integral.add_sample(res * cont_weight);
            } else {
                unreachable!()
            }
        }

        grid.update(1.5, 100);

        integral.update_iter();

        println!("Integral: {} +- {}", integral.avg, integral.err);
    }

    #[cfg(feature = "gridplotting")]
    grid.continuous_dimensions[0]
        .plot(&f, "partitioning.svg")
        .unwrap();
}
