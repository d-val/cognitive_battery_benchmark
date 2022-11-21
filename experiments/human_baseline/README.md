# Human Benchmark Experiments
This repo has the code to conduct human benchmarking via
JSPsych, and can be deployed through JATOS on Mechanical Turk.

## Prerequisites
* yarn/npm

## How to use (locally)
1. Clone the repo: `git clone https://github.com/d-val/cognitive_battery_benchmark_jsPsych`
2. Install dependencies: `yarn install`
3. Run experiment locally: `yarn start`
4. Open browser and go to `localhost:80` (or whatever port is defined in terminal)

## How to use (on JATOS)
1. Do steps 1 + 2 from above
2. Build JATOS experiment: `yarn run jatos`
3. Upload generated `.jzip` to JATOS server and setup experiment from there

## TODO:
* Add certificate to server hosting to allow https for EyeTracker