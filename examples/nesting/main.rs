use havana::{AverageAndErrorAccumulator, ContinuousGrid, DiscreteGrid, Grid, Sample};

fn main() {
    let f = |x: f64| (x * std::f64::consts::PI).sin();
    let f1 = |x: f64| x * x;

    let mut disc_grid = DiscreteGrid::new(
        &[2],
        vec![
            Grid::ContinuousGrid(ContinuousGrid::new(1, 10, 1000)),
            Grid::ContinuousGrid(ContinuousGrid::new(1, 10, 1000)),
        ],
        0.01,
    );

    let mut rng = rand::thread_rng();

    let mut integral = AverageAndErrorAccumulator::new();
    let mut samples = vec![Sample::new(); 10000];

    for _ in 1..20 {
        for si in 0..10000 {
            disc_grid.sample(&mut rng, &mut samples[si]);

            if let Sample::DiscreteGrid(weight, xs, cont_sample) = &samples[si] {
                if let Sample::ContinuousGrid(_cont_weight, cs) = &**cont_sample.as_ref().unwrap() {
                    let res = if xs[0] == 0 { f(cs[0]) } else { f1(cs[0]) };
                    disc_grid.add_training_sample(&samples[si], res);
                    integral.add_sample(res * weight, None);
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        }

        disc_grid.update(1.5, 100, true);

        integral.update_iter();

        println!("Integral: {} +- {}", integral.avg, integral.err);
    }

    #[cfg(feature = "gridplotting")]
    disc_grid.child_grids[0].continuous_dimensions[0]
        .plot(&f, "grid_nest_disc1.svg")
        .unwrap();
    #[cfg(feature = "gridplotting")]
    disc_grid.child_grids[1].continuous_dimensions[0]
        .plot(&f1, "grid_nest_disc2.svg")
        .unwrap();
}
