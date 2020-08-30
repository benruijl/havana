use havana::{AverageAndErrorAccumulator, DiscreteGrid, OwnedSample};

fn main() {
    let mut disc_grid = DiscreteGrid::new(&[5], vec![]);

    let mut rng = rand::thread_rng();

    let mut integral = AverageAndErrorAccumulator::new();
    let mut samples = vec![OwnedSample::new(); 10000];

    for _ in 1..20 {
        for si in 0..10000 {
            disc_grid.sample(&mut rng, &mut samples[si]);

            if let OwnedSample::DiscreteGrid(weight, xs, _cont_sample) = &samples[si] {
                let res = match xs[0] {
                    0 => 0.1,
                    1 => 0.2,
                    2 => 0.3,
                    3 => 0.4,
                    4 => 0.5,
                    _ => unreachable!(),
                };
                disc_grid.add_training_sample(&samples[si], res, false);
                integral.add_sample(res * weight);
            } else {
                unreachable!()
            }
        }

        disc_grid.update(1.5, 100);

        integral.update_iter();

        println!("Integral: {} +- {}", integral.avg, integral.err);
    }
}
