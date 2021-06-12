use havana::{AverageAndErrorAccumulator, DiscreteGrid, Sample};

fn main() {
    let mut disc_grid = DiscreteGrid::new(&[5], vec![], 0.01);
    let f = |xs: &[usize]| match xs[0] {
        0 => 0.1,
        1 => 0.2,
        2 => 0.3,
        3 => 0.4,
        4 => 0.5,
        _ => unreachable!(),
    };

    let mut rng = rand::thread_rng();

    let mut integral = AverageAndErrorAccumulator::new();
    let mut samples = vec![Sample::new(); 10000];

    for _ in 1..20 {
        for si in 0..10000 {
            disc_grid.sample(&mut rng, &mut samples[si]);

            if let Sample::DiscreteGrid(weight, xs, _cont_sample) = &samples[si] {
                let res = f(xs);
                disc_grid.add_training_sample(&samples[si], res, false);
                integral.add_sample(res * weight, &samples[si]);
            } else {
                unreachable!()
            }
        }

        disc_grid.update(1.5, 100);

        integral.update_iter();

        println!("Integral: {} +- {}", integral.avg, integral.err);
    }

    #[cfg(feature = "gridplotting")]
    disc_grid.discrete_dimensions[0]
        .plot("grid_disc.svg")
        .unwrap();
}
