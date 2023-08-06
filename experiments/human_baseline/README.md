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
3. Start an EC2 instances on AWS (or use your own local server/computer), not much compute needed
    * Make sure to allow external access from the internet
4. Install JATOS on EC2 instance, follow instructions [here](https://www.jatos.org/Installation.html)
5. Go to the public address of the EC2 instances and the port specified when you spin up JATOS
6. Click "create a new study" and upload the generated `.jzip` file from step 2
7. Follow setup on JATOS to create a study and run it

## How to use (on Mechanical Turk)
1. On JATOS, go to the study you created and click "Publish"
2. Follow the instructions to create a HIT on Mechanical Turk, and copy the code/URL
3. Go to the Mechanical Turk dashboard and create a HIT
4. Paste the code/URL into the HIT and publish it
5. Go to the HIT and click "Preview" to test it out
6. Once you're satisfied, click "Open" to open the HIT to workers
