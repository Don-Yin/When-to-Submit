# When-to-Submit
This code simulates a manuscript acceptance scoring model reflecting various studies on submission timing, editorial policies, and textual factors.
The submission scores for the year 2025 are precalculated and available at [don-yin.uk/submission](https://don-yin.uk/submission).

[中文](README.md)


## Effects Taken into Considerations
| Study                | Effect                                                     |
|----------------------|------------------------------------------------------------|
| Ausloos et al. 2019a | Seasonal bias in specialized vs interdisciplinary journals |
| Ausloos et al. 2019b | Entropy/diversity indices capture seasonal patterns        |
| Putman et al. 2022   | Weekend submissions more likely desk rejected              |
| Ausloos et al. 2017  | ARCH-like temporal patterns in submissions                 |
| Boja et al. 2018     | Day-of-week affects acceptance likelihood                  |
| Meng et al. 2020     | Turn-of-month surge in accepted papers                     |
| Shalvi et al. 2010   | Summer submissions lower acceptance in psychology          |
| Schreiber 2012       | July highest acceptance rate in physics                    |
| Sweedler 2020        | Short titles and weekday submissions beneficial            |

## Usage
- Use `date_to_score(date_str, model)` to predict a score for a given date
- Run this multiple times to get an average score (due to stochasticity over different types of journals / topics)
- See `year.py` for an example

![Predicted Submission Scores](public/submission_score.png)
(note: higher the better)

## References
1. Ausloos, M., Nedič, O., & Dekanski, A. (2019a). Correlations between submission and acceptance of papers in peer review journals. *Scientometrics*, *119*(1), 279-302. https://doi.org/10.1007/s11192-019-03026-x

2. Ausloos, M., Nedic, O., & Dekanski, A. (2019b). Seasonal entropy, diversity and inequality measures of submitted and accepted papers distributions in peer-reviewed journals. *Entropy*, *21*(6), 564. https://doi.org/10.3390/e21060564

3. Putman, M., Berquist, J. B., Ruderman, E. M., & Sparks, J. A. (2022). Any given Monday: Association between desk rejections and weekend manuscript submissions to rheumatology journals. *The Journal of Rheumatology*, *49*(6), 652-653. https://doi.org/10.3899/jrheum.220099

4. Ausloos, M., Nedic, O., Dekanski, A., Mrowinski, M. J., Fronczak, P., & Fronczak, A. (2017). Day of the week effect in paper submission/acceptance/rejection to/in/by peer review journals. II. An ARCH econometric-like modeling. *Physica A: Statistical Mechanics and its Applications*, *468*, 462-474. https://doi.org/10.1016/j.physa.2016.10.078

5. Boja, C. E., Herţeliu, C., Dârdală, M., & Ileanu, B. V. (2018). Day of the week submission effect for accepted papers in Physica A, PLOS ONE, Nature and Cell. *Scientometrics*, *117*(2), 887-918. https://doi.org/10.1007/s11192-018-2911-7

6. Meng, L., Wang, H., & Han, P. (2020). Getting a head start: Turn-of-the-month submission effect for accepted papers in management journals. *Scientometrics*, *124*(3), 2577-2595. https://doi.org/10.1007/s11192-020-03556-9

7. Shalvi, S., Baas, M., Handgraaf, M. J. J., & De Dreu, C. K. W. (2010). Write when hot — submit when not: Seasonal bias in peer review or acceptance? *Learned Publishing*, *23*(2), 117-123. https://doi.org/10.1087/20100206

8. Schreiber, M. (2012). Seasonal bias in editorial decisions for a physics journal: You should write when you like, but submit in July. *Learned Publishing*, *25*(2), 145-151. https://doi.org/10.1087/20120209

9. Sweedler, J. V. (2020). Strange advice for authors: Submit your manuscript with a short title on a weekday. *Analytical Chemistry*, *92*(3), 2351-2352. https://doi.org/10.1021/acs.analchem.0c00223

# Note
Please refer as: Yin, D. (2024). When-to-Submit. GitHub repository. https://github.com/don-yin/submission-timer
