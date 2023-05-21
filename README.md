# STONKS - Stock Market Forecasting Using News Articles With Reinforcement And Sequence Learning

A final project for MIPT NLP 2023
course https://ods.ai/tracks/nlp-course-spring-23/competitions/final-projects-nlp-course-spring-23

## [Report.pdf](report.pdf)

## Results

| Method Name | MSE     | MAPE | SOAS | S3PS    |
|-------------|---------|------|------|---------|
| RIDGE       | 1253.37 | 0.12 | 0.44 | 4.85%   |
| TREES       | 3919.90 | 0.25 | 0.46 | -14.12% |
| LSTM        | 155.04  | 0.04 | 0.41 | 4.24%   |
| DQNFCN      | 2027.74 | 0.20 | 0.44 | -9.75%  |
| DQNLSTM     | 166.67  | 0.05 | 0.56 | 7.33%   |

![preview](animation.gif)

## Installation

you need python and poetry installed

```shell
make init
make install
```

## Running a notebook

```shell
make notebook
```

## Important scripts and files

1. [dataset notebook](notebooks/dataset.ipynb)
1. [dataset source](data%2Fnews_stocks.csv)
1. [baseline notebook](notebooks/baseline.ipynb)
1. [lstm solution](notebooks%2Fmoods_lstm.ipynb)
1. [dqnlstm solution](notebooks%2Flstm-gym-embeddings.ipynb)

## Contact

[telegram](https://t.me/irusland) & [email](mailto:irusland@icoud.com)

## Credits

[Tinkoff](https://www.tinkoff.ru) &
[Finam](https://www.finam.ru)

## License

MIT License

Copyright (c) 2023 Sirazhetdinov Ruslan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
