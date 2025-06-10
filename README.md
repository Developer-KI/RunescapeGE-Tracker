# RunescapeGET`

https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices
https://runescape.wiki/w/RuneScape:Grand_Exchange_Market_Watch/Usage_and_APIs

https://evidencebasedta.com/montedoc12.15.06.pdf
https://github.com/thegregyang/LossUpAccUp
https://medium.com/@silva.f.francis/avoiding-data-leakage-in-cross-validation-ba344d4d55c0
https://www.robjhyndman.com/papers/mase.pdf

-Factor models
-PCA, correlating PCA factors with HMM states
-HMM against indirect market data, raw prices too dynamic
-Finish XGBoost
-Implement Monte carlo permutations testing
- Expanding window vs time series split
-Mean reversion
-Mean reversion of items against market index (thank you Clay the quant)
-Pipelining market indexes
-Finalizing automated pipelining
-Trading dashboard?
-Tail risk
-Optuna everything
-Outlier detection
-Consider simplifications to classifying price movements vs predictions of price
-Deep dive performance metrics
-Index/pull OSRS updates to inform time series
-Explore technical indicators?
-Research agnostic/uninformed markets, bot trading/activity
-Examine timezone effects
-Research forecasting error criteria


*assymetric MAE
*alpha meta-tuning might cause overfit
*Tons of feature engineering
*Explore more data sources
*Finish backtesting/trading system

*Remember that model selection based on OOS data implicitly causes it to become in sample over time
*Normal k-fold cross validation will leak data (not sure about time series splits?) (group CV)
*Standardizing data before splitting leaks information
*Bootstrapping can help ease sketchy normality assumptions, enables t-tests
*-Covariance matrix estimators/regularization
-Weighing cross-validation split means to emphasize time series splits more recent rather than a flat mean of all MSE/RMSE/etc's over splits
