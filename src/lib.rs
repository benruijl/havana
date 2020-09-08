#[cfg(feature = "gridplotting")]
use plotters::prelude::*;
use rand::Rng;
#[cfg(feature = "use_serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone)]
pub struct AverageAndErrorAccumulator {
    sum: f64,
    sum_sq: f64,
    weight_sum: f64,
    avg_sum: f64,
    pub avg: f64,
    pub err: f64,
    guess: f64,
    pub chi_sq: f64,
    chi_sum: f64,
    chi_sq_sum: f64,
    num_samples: usize,
    pub cur_iter: usize,
}

impl AverageAndErrorAccumulator {
    pub fn new() -> AverageAndErrorAccumulator {
        AverageAndErrorAccumulator {
            sum: 0.,
            sum_sq: 0.,
            weight_sum: 0.,
            avg_sum: 0.,
            avg: 0.,
            err: 0.,
            guess: 0.,
            chi_sq: 0.,
            chi_sum: 0.,
            chi_sq_sum: 0.,
            num_samples: 0,
            cur_iter: 0,
        }
    }

    pub fn add_sample(&mut self, sample: f64) {
        self.sum += sample;
        self.sum_sq += sample * sample;
        self.num_samples += 1;
    }

    pub fn merge_samples(&mut self, other: &mut AverageAndErrorAccumulator) {
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
        self.num_samples += other.num_samples;

        // reset the other
        other.sum = 0.;
        other.sum_sq = 0.;
        other.num_samples = 0;
    }

    pub fn update_iter(&mut self) {
        // TODO: we could be throwing away events that are very rare
        if self.num_samples < 2 {
            self.cur_iter += 1;
            return;
        }

        let n = self.num_samples as f64;
        self.sum /= n;
        self.sum_sq /= n * n;
        let mut w = (self.sum_sq * n).sqrt();

        w = ((w + self.sum) * (w - self.sum)) / (n - 1.);
        if w == 0. {
            w = std::f64::EPSILON;
        }
        w = 1. / w;

        self.weight_sum += w;
        self.avg_sum += w * self.sum;
        let sigsq = 1. / self.weight_sum;
        self.avg = sigsq * self.avg_sum;
        self.err = sigsq.sqrt();
        if self.cur_iter == 0 {
            self.guess = self.sum;
        }
        w *= self.sum - self.guess;
        self.chi_sum += w;
        self.chi_sq_sum += w * self.sum;
        self.chi_sq = self.chi_sq_sum - self.avg * self.chi_sum;

        // reset
        self.sum = 0.;
        self.sum_sq = 0.;
        self.num_samples = 0;
        self.cur_iter += 1;
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "use_serde", derive(Serialize, Deserialize))]
pub enum Sample {
    ContinuousGrid(f64, Vec<f64>),
    DiscreteGrid(f64, Vec<usize>, Option<Box<Sample>>),
    MultiChannel(f64, usize, Vec<f64>), // sample point and weights
}

impl Sample {
    pub fn new() -> Sample {
        Sample::ContinuousGrid(0., vec![])
    }

    pub fn get_weight(&self) -> f64 {
        match self {
            Sample::ContinuousGrid(w, _)
            | Sample::DiscreteGrid(w, _, _)
            | Sample::MultiChannel(w, _, _) => *w,
        }
    }

    pub fn to_discrete_grid(&mut self) -> &mut Self {
        match self {
            Sample::DiscreteGrid(..) => {}
            _ => *self = Sample::DiscreteGrid(0., vec![], None),
        }
        self
    }

    pub fn to_continuous_grid(&mut self) -> &mut Self {
        match self {
            Sample::ContinuousGrid(..) => {}
            Sample::MultiChannel(_w, _c, xs) => {
                let mut x = std::mem::take(xs);
                x.clear();
                *self = Sample::ContinuousGrid(0., x);
            }
            Sample::DiscreteGrid(..) => *self = Sample::ContinuousGrid(0., vec![]),
        }
        self
    }
}
#[derive(Debug, Clone)]
pub enum Grid {
    ContinuousGrid(ContinuousGrid),
    DiscreteGrid(DiscreteGrid),
}

impl Grid {
    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R, sample: &mut Sample) {
        match self {
            Grid::ContinuousGrid(g) => g.sample(rng, sample),
            Grid::DiscreteGrid(g) => g.sample(rng, sample),
        }
    }

    pub fn add_training_sample(&mut self, sample: &Sample, fx: f64, train_on_avg: bool) {
        match self {
            Grid::ContinuousGrid(g) => g.add_training_sample(sample, fx, train_on_avg),
            Grid::DiscreteGrid(g) => g.add_training_sample(sample, fx, train_on_avg),
        }
    }

    pub fn update<'a>(&mut self, alpha: f64, new_bin_length: usize) {
        match self {
            Grid::ContinuousGrid(g) => g.update(alpha, new_bin_length),
            Grid::DiscreteGrid(g) => g.update(alpha, new_bin_length),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscreteDimension {
    pub cdf: Vec<f64>,
    bin_importance: Vec<f64>,
    counter: Vec<usize>,
    min_probability_per_bin: f64,
}

impl DiscreteDimension {
    pub fn new(n_values: usize, min_probability_per_bin: f64) -> DiscreteDimension {
        DiscreteDimension {
            cdf: (1..=n_values).map(|i| i as f64 / n_values as f64).collect(),
            bin_importance: vec![0.; n_values],
            counter: vec![0; n_values],
            min_probability_per_bin,
        }
    }

    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> (usize, f64) {
        let r: f64 = rng.gen();

        let mut bottom = 0.;
        for (i, v) in self.cdf.iter().enumerate() {
            if r <= *v {
                return (i, 1. / (*v - bottom));
            }
            bottom = *v;
        }
        unreachable!("Could not sample discrete dimension: {:?}", self.cdf);
    }

    fn add_training_sample(&mut self, sample: usize, weight: f64, fx: f64, train_on_avg: bool) {
        let prob = if sample > 0 {
            self.cdf[sample] - self.cdf[sample - 1]
        } else {
            self.cdf[sample]
        };

        if train_on_avg {
            self.bin_importance[sample] += weight * fx * prob;
        } else {
            self.bin_importance[sample] += weight * weight * fx * fx * prob * prob;
        }

        self.counter[sample] += 1;
    }

    fn update<'a>(&mut self, _alpha: f64) {
        if self.bin_importance.iter().all(|x| *x == 0.) {
            return;
        }

        for avg in self.bin_importance.iter_mut() {
            *avg = avg.abs();
        }

        let mut sum = 0.;
        for (avg, &c) in self.bin_importance.iter_mut().zip(&self.counter) {
            if c > 0 {
                *avg /= c as f64;
                sum += *avg;
            }
        }

        // TODO: factor in previous cdf
        let mut accum = 0.;
        for (c, &a) in self.cdf.iter_mut().zip(&self.bin_importance) {
            accum += a / sum * (1. - self.min_probability_per_bin)
                + self.min_probability_per_bin / self.bin_importance.len() as f64;
            *c = accum;
        }

        self.counter.clear();
        self.counter.resize(self.cdf.len(), 0);
        self.bin_importance.clear();
        self.bin_importance.resize(self.cdf.len(), 0.);
    }

    #[cfg(feature = "gridplotting")]
    pub fn plot(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let max_prob = self
            .cdf
            .iter()
            .fold((0.0f64, 0.0f64), |(last, max), x| (*x, max.max(x - last)))
            .1;

        let root = SVGBackend::new(filename, (640 * (self.cdf.len() as u32 / 10), 640))
            .into_drawing_area();
        root.fill(&WHITE)?;
        let root = root.margin(10, 10, 10, 10);
        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_ranged(0u32..(self.cdf.len() as u32), 0f32..max_prob as f32)?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .line_style_1(&WHITE.mix(0.3))
            .x_labels(self.cdf.len())
            .y_labels(10)
            .draw()?;

        let mut bottom = 0.;
        chart
            .draw_series(
                plotters::series::Histogram::vertical(&chart)
                    .style(RED.mix(0.8).filled())
                    .data(self.cdf.iter().enumerate().map(|(i, x)| {
                        let r = (i as u32, (*x - bottom) as f32);
                        bottom = *x;
                        r
                    })),
            )
            .unwrap();

        Ok(())
    }
}

// TODO: support a mix of continuous dimensions and discrete dimensions?
#[derive(Debug, Clone)]
pub struct DiscreteGrid {
    pub discrete_dimensions: Vec<DiscreteDimension>,
    pub child_grids: Vec<Grid>,
    accumulator: AverageAndErrorAccumulator,
}

impl DiscreteGrid {
    pub fn new(
        values_per_dim: &[usize],
        child_grids: Vec<Grid>,
        minimum_probability: f64,
    ) -> DiscreteGrid {
        DiscreteGrid {
            discrete_dimensions: values_per_dim
                .iter()
                .map(|i| DiscreteDimension::new(*i, minimum_probability))
                .collect(),
            child_grids,
            accumulator: AverageAndErrorAccumulator::new(),
        }
    }

    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R, sample: &mut Sample) {
        if let Sample::DiscreteGrid(weight, vs, child) = sample.to_discrete_grid() {
            *weight = 1.;

            vs.clear();
            vs.resize(self.discrete_dimensions.len(), 0);

            // FIXME: the child index is wrong beyond one dimension!
            let mut child_index = 0;
            for (vs, d) in vs.iter_mut().zip(&self.discrete_dimensions) {
                let (v, w) = d.sample(rng);
                *weight *= w;
                *vs = v;

                if self.discrete_dimensions.len() > 1 {
                    panic!("Indexing not generic yet");
                }

                child_index += v; // (v as f64 * d.cdf.len() as f64) as usize;
            }

            // get the child grid for this sample
            if child_index < self.child_grids.len() {
                let child_sample = if let Some(child_sample) = child {
                    child_sample
                } else {
                    *child = Some(Box::new(Sample::new()));
                    child.as_mut().unwrap()
                };

                self.child_grids[child_index].sample(rng, child_sample);

                // multiply the weight of the subsample
                *weight *= child_sample.get_weight();
            } else {
                *child = None;
            };
        } else {
            unreachable!("Sample cannot be converted to discrete sample: {:?}", self);
        }
    }

    pub fn add_training_sample(&mut self, sample: &Sample, fx: f64, train_on_avg: bool) {
        if let Sample::DiscreteGrid(weight, xs, sub_sample) = sample {
            let mut child_index = 0;
            for (d, sdim) in self.discrete_dimensions.iter_mut().zip(xs) {
                child_index += *sdim; // (*sdim as f64 * d.cdf.len() as f64) as usize;
                d.add_training_sample(*sdim, *weight, fx, train_on_avg)
            }

            if let Some(s) = sub_sample {
                self.child_grids[child_index].add_training_sample(&*s, fx, train_on_avg);
            }

            self.accumulator.add_sample(fx * weight);
        } else {
            unreachable!("Sample cannot be converted to discrete sample: {:?}", self);
        }
    }

    pub fn update(&mut self, alpha: f64, new_bin_length: usize) {
        for d in self.discrete_dimensions.iter_mut() {
            d.update(alpha);
        }

        for d in self.child_grids.iter_mut() {
            d.update(alpha, new_bin_length);
        }

        self.accumulator.update_iter();
        /*println!(
            "Discrete grid result: {} +- {}",
            self.accumulator.avg, self.accumulator.err
        );*/
    }
}

#[derive(Debug, Clone)]
pub struct ContinuousGrid {
    pub continuous_dimensions: Vec<ContinuousDimension>,
    accumulator: AverageAndErrorAccumulator,
}

impl ContinuousGrid {
    pub fn new(n_dims: usize, n_bins: usize, min_samples_for_update: usize) -> ContinuousGrid {
        ContinuousGrid {
            continuous_dimensions: vec![
                ContinuousDimension::new(n_bins, min_samples_for_update);
                n_dims
            ],
            accumulator: AverageAndErrorAccumulator::new(),
        }
    }

    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R, sample: &mut Sample) {
        if let Sample::ContinuousGrid(weight, vs) = sample.to_continuous_grid() {
            *weight = 1.;
            vs.clear();
            vs.resize(self.continuous_dimensions.len(), 0.);
            for (vs, d) in vs.iter_mut().zip(&self.continuous_dimensions) {
                let (v, w) = d.sample(rng);
                *weight *= w;
                *vs = v;
            }
        } else {
            unreachable!(
                "Sample cannot be converted to continuous sample: {:?}",
                self
            );
        }
    }

    pub fn add_training_sample(&mut self, sample: &Sample, fx: f64, train_on_avg: bool) {
        if let Sample::ContinuousGrid(weight, xs) = sample {
            self.accumulator.add_sample(fx * weight);

            for (d, sdim) in self.continuous_dimensions.iter_mut().zip(xs) {
                d.add_training_sample(*sdim, *weight, fx, train_on_avg)
            }
        } else {
            unreachable!(
                "Sample cannot be converted to continuous sample: {:?}",
                self
            );
        }
    }

    pub fn update(&mut self, alpha: f64, new_bin_length: usize) {
        for d in self.continuous_dimensions.iter_mut() {
            d.update(alpha, new_bin_length);
        }

        self.accumulator.update_iter();
        /*println!(
            "Result: {} +- {}",
            self.accumulator.avg, self.accumulator.err
        );*/
    }
}

#[derive(Debug, Clone)]
pub struct ContinuousDimension {
    pub partitioning: Vec<f64>,
    pub new_partitioning: Vec<f64>,
    bin_importance: Vec<f64>,
    counter: Vec<usize>,
    min_samples_for_update: usize,
}

impl ContinuousDimension {
    pub fn new(n_bins: usize, min_samples_for_update: usize) -> ContinuousDimension {
        ContinuousDimension {
            partitioning: (0..=n_bins).map(|i| i as f64 / n_bins as f64).collect(),
            new_partitioning: vec![],
            bin_importance: vec![0.; n_bins],
            counter: vec![0; n_bins],
            min_samples_for_update,
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> (f64, f64) {
        let r: f64 = rng.gen();

        // map the point to a bin
        let n_bins = self.partitioning.len() - 1;
        let bin_index = (n_bins as f64 * r) as usize;
        let bin_width = self.partitioning[bin_index + 1] - self.partitioning[bin_index];

        // rescale the point in the bin
        let sample =
            self.partitioning[bin_index] + (n_bins as f64 * r - bin_index as f64) * bin_width;
        let weight = n_bins as f64 * bin_width; // d_sample / d_r

        (sample, weight)
    }

    fn add_training_sample(&mut self, sample: f64, weight: f64, fx: f64, train_on_avg: bool) {
        let mut index = self
            .partitioning
            .binary_search_by(|v| v.partial_cmp(&sample).unwrap())
            .unwrap_or_else(|e| e);
        if index > 0 {
            index -= 1;
        }

        if train_on_avg {
            self.bin_importance[index] += weight * fx;
        } else {
            self.bin_importance[index] += weight * weight * fx * fx;
        }

        self.counter[index] += 1;
    }

    fn update<'a>(&mut self, alpha: f64, new_bin_length: usize) {
        if self.counter.iter().sum::<usize>() < self.min_samples_for_update {
            // do not train the grid if there is a lack of samples
            return;
        }

        if alpha == 0. {
            self.bin_importance.clear();
            self.bin_importance.resize(self.partitioning.len() - 1, 0.);
            self.counter.clear();
            self.counter.resize(self.partitioning.len() - 1, 0);
            return;
        }

        let n_bins = self.partitioning.len() - 1;

        //dbg!(&self.bin_importance, &self.counter);

        for avg in self.bin_importance.iter_mut() {
            *avg = avg.abs();
        }

        // normalize the average
        for (avg, &c) in self.bin_importance.iter_mut().zip(&self.counter) {
            if c > 0 {
                *avg /= c as f64;
            }
        }

        // smoothen the averages between adjacent grid points
        // TODO: tune if/how this is done?
        // Cuba takes a simple average
        if self.partitioning.len() > 2 {
            let mut prev = self.bin_importance[0];
            let mut cur = self.bin_importance[1];
            self.bin_importance[0] = (3. * prev + cur) / 4.;
            for bin in 1..n_bins - 1 {
                let s = prev + cur * 6.;
                prev = cur;
                cur = self.bin_importance[bin + 1];
                self.bin_importance[bin] = (s + cur) / 8.;
            }
            self.bin_importance[n_bins - 1] = (prev + 3. * cur) / 4.;
        }

        // our sum is positive definite, so it's only zero when everything is 0
        let sum: f64 = self.bin_importance.iter().sum();
        let mut impsum = 0.;
        for bi in self.bin_importance.iter_mut() {
            let m = if *bi == 0. {
                0.
            } else if *bi == sum {
                1.
            } else {
                ((*bi / sum - 1.) / (*bi / sum).ln()).powf(alpha)
            };
            *bi = m;
            impsum += m;
        }

        //dbg!(&self.bin_importance);

        let new_weight_per_bin = impsum / new_bin_length as f64;
        //dbg!(new_weight_per_bin);

        // resize the bins using their importance measure
        self.new_partitioning.clear();
        self.new_partitioning.resize(new_bin_length + 1, 0.);

        // evenly distribute the bins such that each has weight_per_bin weight
        let mut acc = 0.;
        let mut j = 0;
        let mut target = 0.;
        for nb in &mut self.new_partitioning[1..].iter_mut() {
            target += new_weight_per_bin;
            // find the bin that has the accumulated weight we are looking for
            // FIXME: sometimes we get an infinite loop
            while j < self.bin_importance.len() && acc + self.bin_importance[j] < target {
                acc += self.bin_importance[j];
                // prevent some rounding errors from going out of the bin
                if j + 1 < self.bin_importance.len() {
                    j += 1;
                } else {
                    break;
                }
            }

            // find out how deep we are in the current bin
            let bin_depth = (target - acc) / self.bin_importance[j];
            *nb = self.partitioning[j]
                + bin_depth * (self.partitioning[j + 1] - self.partitioning[j]);
        }

        // it could be that all the weights are distributed before we reach 1, for example if the first bin
        // has all the weights. we still force to have the complete input range
        self.new_partitioning[new_bin_length] = 1.0;
        std::mem::swap(&mut self.partitioning, &mut self.new_partitioning);

        self.bin_importance.clear();
        self.bin_importance.resize(self.partitioning.len() - 1, 0.);
        self.counter.clear();
        self.counter.resize(self.partitioning.len() - 1, 0);
    }

    #[cfg(feature = "gridplotting")]
    pub fn plot(
        &self,
        f: &dyn Fn(f64) -> f64,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let f_evals: Vec<(f32, f32)> = (0..101)
            .map(|i| (i as f32 / 100., f(i as f64 / 100.) as f32))
            .collect();
        let has_negative = f_evals.iter().any(|e| e.1 < 0.);

        let root = SVGBackend::new(filename, (640, 640)).into_drawing_area();
        root.fill(&WHITE)?;
        let root = root.margin(10, 10, 10, 10);
        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_ranged(
                0f32..1f32,
                if has_negative {
                    -1f32..1f32
                } else {
                    0f32..1f32
                },
            )?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .x_labels(10)
            .y_labels(10)
            //.y_label_formatter(&|x| format!("{:.2}", x))
            .draw()?;

        chart.draw_series(LineSeries::new(f_evals.into_iter(), &RED))?;

        for p in &self.partitioning {
            let test = (0..2).map(|i| {
                (
                    *p as f32,
                    if has_negative {
                        i as f32 * 2. - 1.
                    } else {
                        i as f32
                    },
                )
            });
            chart.draw_series(LineSeries::new(test, &BLACK))?;
        }

        Ok(())
    }
}
