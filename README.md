# Systematic human learning and generalization from a brief tutorial with explanatory feedback.

Publicly available code and data.

## Running the experiment

### Settings
Most settings can be found in `experiment/src/app/AppSettings.ts`. Most important are 
`testEnvironment`, `hostURL`, `postURL`, and `saveDirectory`.

### Puzzle Server
Puzzles are randomly generated from a Python Flask server and may be accessed via HTTP.
The React application queries the server for puzzles, so the Flask server must be up before
running the React application. To deploy the server, run

```
python python/scripts/rest_api_endpoint.py
```

You can see the generated puzzles and experiment details at `http://localhost:5001/sudoku_hs/experiment/new`.


### Local deployment
The experiment was written using React and may be executed locally.
First, check that `hostURL` in `AppSettings.ts` is set to wherever the Flask server is deployed, e.g.
`http://someurl:5001` or `http://localhost:5001`.

Then go into `hidden_singles_public/experiment` and run

```
npm install
npm start
```

If `testEnvironment` is true, `workerId` and `skipTo` may be added as additional parameters to the URL, such as `localhost:3000/?workerId=Andrew&skipTo=30`.
`workerId` sets the seed for the randomized puzzles. `skipTo` allows skipping to different screens in the experiment.

### Deployment

This experiment was originally deployed on Amazon MTurk using Psiturk. I no longer recommend using MTurk to conduct online experiments and use Prolific instead. This section is included for completeness, not for future replications.

To deploy on Amazon MTurk, it is helpful to have a dedicated server, such as AWS.

First, clone into the server, then build and run the project using 

```
npm install
npm run build
serve -s build
```

Next, install PsiTurk from [PsiTurk](http://psiturk.org/). Before deployment, be sure to check `psiturk/config.txt`.
To deploy, go into the `psiturk/` directory and run

```
psiturk
```

You should enter the PsiTurk program where you can execute PsiTurk commands.

### Saving data

The React app will periodically POST data to `postURL` in `AppSettings.ts`. This needs to point to any server that can receive POST requests.
The POST payload is a JSON stream with two fields: `filename` and `data`. If you have your own server that can handle POST requests, you can handle these
data however you wish.

In our experiments, we set up a cgi-bin with Python code to handle the incoming data. See `cgi-bin/post-endpoint.py`.



## Data

The data from our experiments are organized as such:

### Participant Data

- `data/processed/puzzle_data.tsv`: contains trial-level data for each puzzle completed by each participant
- `data/processed/subject_data.tsv`: contains participant-level data including overall performance, survey results, etc.
- `data/processed/qrater.tsv`: contains data provided to the questionnaire raters
- `data/processed/qrater.tsv`: same as qrater.tsv, except just the first 20 participants

### Questionnaire Ratings

- `data/qratings/rater1.tsv`: contains questionnaire ratings from Rater 1
- `data/qratings/rater2.tsv`: contains questionnaire ratings from Rater 2

### Hidden Markov Model

`{group}` refers to either solvers or nonsolvers.

- `data/hmm/{group}/nll_actual.tsv`: contains negative log-likelihood for participants' actual data evaluated using fitted HMMs
- `data/hmm/{group}/nll_sample.tsv`: contains negative log-likelihood for sampled data from fitted HMM, evaluated using the fitted HMMs
- `data/hmm/{group}/subject_p_responses.tsv`: contains P(response_t | data) for each trial for each participant
- `data/hmm/{group}/subject_p_strategies.tsv`: contains P(strategy_t | data) for each trial for each participant
- `data/hmm/{group}/top_paths.tsv`: contains top 5 strategy paths for each participant based on the posterior probabilities
- `data/hmm/{group}/top_paths_probs.tsv`: contains the posterior probability P(strategy_t | data) for the top 5 strategy paths for each participant

### Recurrent Relational Network

- `data/rrn_dataset/train.csv`: original training data used in Palm et al. (2016)
- `data/rrn_dataset/valid.csv`: original validation data used in Palm et al. (2016)
- `data/rrn_dataset/test.csv`: original test data used in Palm et al. (2016)
- `data/rrn_dataset/2x3_puzzles.tsv`: Sudoku puzzles with 36 cells. Useful for testing model with simpler puzzles.
- `data/rrn/drrn_test_results.tsv`: Digit-Invariant RRN results for out-of-distribution Hidden Single test puzzles
- `data/rrn/drrn_train_results.tsv`: log of metrics for Digit-Invariant RRN while training (e.g. loss, accuracy)
- `data/rrn/rrn_test_results.tsv`: RRN results for out-of-distribution Hidden Single test puzzles
- `data/rrn/rrn_train_results.tsv`: log of metrics for RRN while training (e.g. loss, accuracy)
- `data/rrn/sudoku_2x3_results.tsv`: log of metrics for RRN while training on 36-cell Sudoku puzzles
- `data/rrn/sudoku_3x3_results.tsv`: log of metrics for RRN while training on 81-cell Sudoku puzzles

### Data Wrangling

To process the participant data files after running the experiment, first store all participants' generated data into 
`data/raw`. Then run through the `python/scripts/Data Wrangler.ipynb` iPython Notebook.
Next, open `r/hidden_singles_public.Rproj` and run through the `data_wrangler.Rmd` R Markdown file.
Note that the Rmd is not idempotent and should not be run more than once.
To regenerate the data, run both the `ipynb` and `Rmd` each exactly once in that order.


## Computational Models

All computational models were written in Python and can be run by executing iPython Notebooks.

- `python/scripts/Hidden Markov Model`: code for training HMM to Practice Phase data. Saves output files to `data/hmm/`
- `python/scripts/RRN - Sudoku`: code for replicating Palm et al. (2016) results. Saves output files to `data/rrn/`
- `python/scripts/RRN - Sudoku 2x3`: code for replicating Palm et al. (2016) using simpler 36-cell puzzles. Saves output files to `data/rrn/`
- `python/scripts/RRN - Hidden Singles`: code for training and evaluating RRN and Digit-Invariant RRN to Hidden Singles puzzles. Saves output files to `data/rrn/`



## Analyses and Figure Generation

### R Files
Analyses and data visualizations were done in R and can be run by executing R Markdown files.
`.Rmd` files should be run within the `r/hidden_singles_public.Rproj` context.

- `practice_phase.Rmd`: code for analyzing Practice Phase data.
- `questionnaire.Rmd`: code for analyzing Questionnaire data, including education and free-response ratings.
- `recurrent_relational_network.Rmd`: code for analyzing RRN results.
- `test_phase.Rmd`: code for analyzing Test Phase data. Trains BRMS models that are cached into `r/cache/`.

### Python Files
Some figures were generated using Python.

- `python/scripts/Puzzle Figure Generator.ipynb`: code for generating Hidden Singles puzzle figures.
- `Screenshot Generator.ipynb`: code for taking screenshots of the React application. The app and Flask server must both be running locally.


## Figures

All figures included in the paper, as well as ones not included, can be found in the `figures` directory.

## Other documents

- `documents/Sudoku Rater Instructions.pdf`: instructions given the Questionnaire Raters.
- `data/test_phase_coefficients.tsv`: Test Phase regression coefficients
- `r/questionnaire.pdf`: contains results for education regressions, including coefficients, p-values, etc.
