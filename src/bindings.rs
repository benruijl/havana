use crate::*;
use pyo3::{prelude::*, types::PyBytes};
use rand::SeedableRng;
use tinymt::{TinyMT64, TinyMT64Seed};

#[pyclass(name = "Havana")]
pub struct HavanaWrapper {
    pub grid: Grid,
    rng: TinyMT64,
    pub samples: Vec<Sample>,
}

#[pyclass(name = "Sample")]
pub struct SampleWrapper {
    #[pyo3(get)]
    discrete_sample: Vec<(f64, Vec<usize>)>,
    #[pyo3(get)]
    continuous_sample: Vec<(f64, Vec<f64>)>,
}

#[pyclass]
#[derive(Clone)]
pub struct GridConstructor {
    continuous_grid_constructor: Option<ContinuousGridConstructor>,
    discrete_grid_constructor: Option<DiscreteGridConstructor>,
}

#[pymethods]
impl GridConstructor {
    #[new]
    fn new(cgc: Option<ContinuousGridConstructor>, dgc: Option<DiscreteGridConstructor>) -> Self {
        GridConstructor {
            continuous_grid_constructor: cgc,
            discrete_grid_constructor: dgc,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ContinuousGridConstructor {
    n_dims: usize,
    n_bins: usize,
    min_samples_for_update: usize,
}

#[pymethods]
impl ContinuousGridConstructor {
    #[new]
    fn new(n_dims: usize, n_bins: usize, min_samples_for_update: usize) -> Self {
        ContinuousGridConstructor {
            n_dims,
            n_bins,
            min_samples_for_update,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct DiscreteGridConstructor {
    values_per_dim: Vec<usize>,
    child_grids: Vec<GridConstructor>,
    max_prob_ratio: f64,
}

#[pymethods]
impl DiscreteGridConstructor {
    #[new]
    fn new(
        values_per_dim: Vec<usize>,
        child_grids: Vec<GridConstructor>,
        max_prob_ratio: f64,
    ) -> Self {
        DiscreteGridConstructor {
            values_per_dim,
            child_grids,
            max_prob_ratio,
        }
    }
}

impl HavanaWrapper {
    fn construct_grid(g: &GridConstructor) -> Grid {
        if let Some(cg) = &g.continuous_grid_constructor {
            Grid::ContinuousGrid(ContinuousGrid::new(
                cg.n_dims,
                cg.n_bins,
                cg.min_samples_for_update,
            ))
        } else if let Some(dg) = &g.discrete_grid_constructor {
            Grid::DiscreteGrid(DiscreteGrid::new(
                &dg.values_per_dim,
                dg.child_grids
                    .iter()
                    .map(|chg| HavanaWrapper::construct_grid(chg))
                    .collect::<Vec<_>>(),
                dg.max_prob_ratio,
            ))
        } else {
            unreachable!("No grid specified")
        }
    }
}

#[pymethods]
impl HavanaWrapper {
    #[new]
    fn new(g: GridConstructor, seed: Option<u64>) -> HavanaWrapper {
        HavanaWrapper {
            grid: HavanaWrapper::construct_grid(&g),
            rng: if let Some(seed) = seed {
                TinyMT64::from_seed(TinyMT64Seed::from(seed))
            } else {
                TinyMT64::from_entropy()
            },
            samples: vec![],
        }
    }

    #[staticmethod]
    #[args(format = "\"yaml\"", seed = "None")]
    fn load_grid(data: &[u8], format: &str, seed: Option<u64>) -> PyResult<Self> {
        let grid = match format {
            "yaml" => serde_yaml::from_slice(data)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?,
            "bin" => bincode::deserialize(data)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?,
            _ => return Err(pyo3::exceptions::PyIOError::new_err("Unknown format")),
        };

        Ok(HavanaWrapper {
            grid,
            rng: if let Some(seed) = seed {
                TinyMT64::from_seed(TinyMT64Seed::from(seed))
            } else {
                TinyMT64::from_entropy()
            },
            samples: vec![],
        })
    }

    #[args(format = "\"yaml\"")]
    fn dump_grid<'p>(&self, py: Python<'p>, format: &str) -> PyResult<&'p PyBytes> {
        match format {
            "yaml" => serde_yaml::to_vec(&self.grid)
                .map(|a| PyBytes::new(py, &a))
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string())),
            "bin" => bincode::serialize(&self.grid)
                .map(|a| PyBytes::new(py, &a))
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string())),
            _ => return Err(pyo3::exceptions::PyTypeError::new_err("Unknown format")),
        }
    }

    fn sample(&mut self, num_samples: usize) -> PyResult<()> {
        self.samples.resize(num_samples, Sample::new());
        for s in &mut self.samples {
            self.grid.sample(&mut self.rng, s);
        }

        Ok(())
    }

    /// Get the samples in Python. This involves a slow conversion.
    fn get_samples(&self) -> PyResult<Vec<SampleWrapper>> {
        let mut out_samples = Vec::with_capacity(self.samples.len());
        for s in &self.samples {
            let mut sample_wrapper = SampleWrapper {
                discrete_sample: vec![],
                continuous_sample: vec![],
            };

            let mut cur_s = Some(&*s);
            while let Some(cs) = cur_s {
                match cs {
                    Sample::ContinuousGrid(w, x) => {
                        sample_wrapper.continuous_sample.push((*w, x.clone()));
                        break;
                    }
                    Sample::DiscreteGrid(w, x, sub_sample) => {
                        sample_wrapper.discrete_sample.push((*w, x.to_vec()));
                        cur_s = sub_sample.as_ref().map(|x| &**x);
                    }
                    Sample::MultiChannel(_, _, _) => unimplemented!(),
                }
            }

            out_samples.push(sample_wrapper);
        }

        Ok(out_samples)
    }

    fn add_training_samples(&mut self, fx: Vec<f64>) -> PyResult<()> {
        if fx.len() != self.samples.len() {
            return PyResult::Err(pyo3::exceptions::PyAssertionError::new_err(
                "Number of returned values does not equal number of samples",
            ));
        }

        for (s, f) in self.samples.iter().zip(&fx) {
            self.grid.add_training_sample(s, *f);
        }

        Ok(())
    }

    fn merge(&mut self, other: &HavanaWrapper) -> PyResult<()> {
        self.grid.merge(&other.grid);
        Ok(())
    }

    fn update(
        &mut self,
        alpha: f64,
        new_bin_length: usize,
        train_on_avg: bool,
    ) -> PyResult<(f64, f64, f64)> {
        self.grid.update(alpha, new_bin_length, train_on_avg);

        let acc = match &self.grid {
            Grid::ContinuousGrid(cs) => &cs.accumulator,
            Grid::DiscreteGrid(ds) => &ds.accumulator,
        };

        Ok((acc.avg, acc.err, acc.chi_sq))
    }

    fn get_top_level_accumulators(&self) -> PyResult<Vec<(f64, f64, f64, f64, f64, usize, usize)>> {
        match &self.grid {
            Grid::ContinuousGrid(cs) => Ok(vec![(
                cs.accumulator.avg,
                cs.accumulator.err,
                cs.accumulator.chi_sq,
                cs.accumulator.max_eval_negative,
                cs.accumulator.max_eval_positive,
                cs.accumulator.processed_samples,
                cs.accumulator.num_zero_evals,
            )]),
            Grid::DiscreteGrid(ds) => Ok(ds.discrete_dimensions[0]
                .bin_accumulator
                .iter()
                .map(|a| {
                    (
                        a.avg,
                        a.err,
                        a.chi_sq,
                        a.max_eval_negative,
                        a.max_eval_positive,
                        a.processed_samples,
                        a.num_zero_evals,
                    )
                })
                .collect::<Vec<_>>()),
        }
    }

    fn get_top_level_cdfs(&self) -> PyResult<Option<Vec<f64>>> {
        match &self.grid {
            Grid::ContinuousGrid(_cs) => Ok(None),
            Grid::DiscreteGrid(ds) => Ok(Some(ds.discrete_dimensions[0].cdf.clone())),
        }
    }

    fn get_current_estimate(&self) -> PyResult<(f64, f64, f64, f64, f64, usize, usize)> {
        match &self.grid {
            Grid::ContinuousGrid(cs) => {
                let mut a = cs.accumulator.shallow_copy();
                a.update_iter();
                Ok((
                    a.avg,
                    a.err,
                    a.chi_sq,
                    a.max_eval_negative,
                    a.max_eval_positive,
                    a.processed_samples,
                    a.num_zero_evals,
                ))
            }
            Grid::DiscreteGrid(ds) => {
                let mut a = ds.accumulator.shallow_copy();
                a.update_iter();
                Ok((
                    a.avg,
                    a.err,
                    a.chi_sq,
                    a.max_eval_negative,
                    a.max_eval_positive,
                    a.processed_samples,
                    a.num_zero_evals,
                ))
            }
        }
    }
}

#[pymodule]
fn havana(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HavanaWrapper>()?;
    m.add_class::<SampleWrapper>()?;
    m.add_class::<GridConstructor>()?;
    m.add_class::<ContinuousGridConstructor>()?;
    m.add_class::<DiscreteGridConstructor>()?;

    Ok(())
}
