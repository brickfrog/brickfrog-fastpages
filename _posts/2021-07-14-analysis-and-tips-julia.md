---
title: "Numerai Tips & Tricks in Julia!"
description: "This post is a (near) 1:1 replication of the Python tips & tricks for the Numerai data science competition."
layout: post
toc: false
comments: true
image: images/numerai.png
hide: false
search_exclude: true
categories: [numerai, MLJ, scikit-learn, datascience, julia]
hide_binder_badge: true
hide_colab_badge: true
---

```julia
using DataFrames # Requires > 0.22.0 for rownums function
using CSV
using Statistics
using LinearAlgebra
using Plots
using StatsBase
using Distributions
using MLJ
using MLJLinearModels
using MLJXGBoostInterface
using XGBoost
import MLJBase: train_test_pairs

# Using for Logistic + CV options, also as an example of how to use Sklearn within Julia
using ScikitLearn

@sk_import linear_model: (LogisticRegression, LinearRegression)
@sk_import model_selection: (TimeSeriesSplit, KFold, GroupKFold, cross_val_score)
@sk_import metrics: make_scorer
```

    ┌ Warning: Module model_selection has been ported to Julia - try `import ScikitLearn: CrossValidation` instead
    └ @ ScikitLearn.Skcore C:\Users\Justin\.julia\packages\ScikitLearn\ssekP\src\Skcore.jl:179
    




    PyObject <function make_scorer at 0x000000008F07CA60>




```julia
df = CSV.File("numerai_training_data.csv") |> DataFrame

first(df,5)
```




<div class="data-frame"><p>5 rows × 314 columns (omitted printing of 309 columns)</p><table class="data-frame"><thead><tr><th></th><th>id</th><th>era</th><th>data_type</th><th>feature_intelligence1</th><th>feature_intelligence2</th></tr><tr><th></th><th title="String">String</th><th title="String">String</th><th title="String">String</th><th title="Float64">Float64</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>n000315175b67977</td><td>era1</td><td>train</td><td>0.0</td><td>0.5</td></tr><tr><th>2</th><td>n0014af834a96cdd</td><td>era1</td><td>train</td><td>0.0</td><td>0.0</td></tr><tr><th>3</th><td>n001c93979ac41d4</td><td>era1</td><td>train</td><td>0.25</td><td>0.5</td></tr><tr><th>4</th><td>n0034e4143f22a13</td><td>era1</td><td>train</td><td>1.0</td><td>0.0</td></tr><tr><th>5</th><td>n00679d1a636062f</td><td>era1</td><td>train</td><td>0.25</td><td>0.25</td></tr></tbody></table></div>




```julia
# There are 501808 rows grouped into eras, and a single target (target)

size(df)
```




    (501808, 314)




```julia
features = select(df, r"feature") |> names
df.erano = parse.(Int64, replace.(df.era, "era" => ""))
eras = df.erano
target = "target"
length(features)
```




    310




```julia
# The features are grouped together into 6 types
feature_groups =
    Dict(g => [c for c in features if startswith(c, "feature_$g")] 
        for g in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"])
```




    Dict{String, Vector{String}} with 6 entries:
      "charisma"     => ["feature_charisma1", "feature_charisma2", "feature_charism…
      "constitution" => ["feature_constitution1", "feature_constitution2", "feature…
      "dexterity"    => ["feature_dexterity1", "feature_dexterity2", "feature_dexte…
      "wisdom"       => ["feature_wisdom1", "feature_wisdom2", "feature_wisdom3", "…
      "strength"     => ["feature_strength1", "feature_strength2", "feature_strengt…
      "intelligence" => ["feature_intelligence1", "feature_intelligence2", "feature…




```julia
# The models should be scored based on the rank-correlation (spearman) with the target
# There's probably (definitely) a better way to write this - [ordinalrank would solve the ranking]

function numerai_score(y_true, y_pred, df)
        rank_pred = sort(combine(groupby(DataFrame(y_pred = y_pred
                                , eras = df.erano
                                , rnum = rownumber.(eachrow(df)))
                        , :eras)
                , sdf -> sort(sdf, :y_pred)
                , :eras => eachindex => :rank
                , nrow => :n)
            , :rnum)

        rank_pred = rank_pred.rank ./ rank_pred.n
    
        cor(y_true, rank_pred)
    end

# It can also be convenient while working to evaluate based on the regular (pearson) correlation
# R2 Score to replicate the Python library outputs

function r2_score(y_true, y_pred)
    @assert length(y_true) == length(y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    mean = sum(y_true) / length(y_true)
    ss_total = sum((y_true .- mean).^2)
    return 1 - ss_res/(ss_total + eps(eltype(y_pred)))
end

# cor() returns a matrix with no need for manipulation, so no need to replicate that here
```




    r2_score (generic function with 1 method)




```julia
# There are 120 eras numbered from 1 to 120

describe(df, :all, cols=:erano)
```




<div class="data-frame"><p>1 rows × 13 columns (omitted printing of 3 columns)</p><table class="data-frame"><thead><tr><th></th><th>variable</th><th>mean</th><th>std</th><th>min</th><th>q25</th><th>median</th><th>q75</th><th>max</th><th>nunique</th><th>nmissing</th></tr><tr><th></th><th title="Symbol">Symbol</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Int64">Int64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Int64">Int64</th><th title="Nothing">Nothing</th><th title="Int64">Int64</th></tr></thead><tbody><tr><th>1</th><td>erano</td><td>64.002</td><td>33.3329</td><td>1</td><td>37.0</td><td>64.0</td><td>93.0</td><td>120</td><td></td><td>0</td></tr></tbody></table></div>




```julia
# The earlier eras are smaller, but generally each era is 4000-5000 rows

group_df = combine(groupby(df, :erano), nrow => :count)
plot(group_df.erano, group_df.count)
```




    
![svg](_posts/mdimages/analysis_and_tips_julia_7_0.svg)
    



    WARNING: CPU random generator seem to be failing, disabling hardware random number generation
    WARNING: RDRND generated: 0xffffffff 0xffffffff 0xffffffff 0xffffffff
    


```julia
# The target is discrete and takes on 5 different values

combine(groupby(df, :target), nrow => :count)
```




<div class="data-frame"><p>5 rows × 2 columns</p><table class="data-frame"><thead><tr><th></th><th>target</th><th>count</th></tr><tr><th></th><th title="Float64">Float64</th><th title="Int64">Int64</th></tr></thead><tbody><tr><th>1</th><td>0.5</td><td>251677</td></tr><tr><th>2</th><td>0.25</td><td>100053</td></tr><tr><th>3</th><td>0.75</td><td>100045</td></tr><tr><th>4</th><td>0.0</td><td>25016</td></tr><tr><th>5</th><td>1.0</td><td>25017</td></tr></tbody></table></div>



# Some of the features are very correlated
Especially within feature groups


```julia
feature_corrs = DataFrame(cor(Matrix(df[!, names(df, features)])), features)
insertcols!(feature_corrs, 1, :features => features)

first(feature_corrs,5)
```




<div class="data-frame"><p>5 rows × 311 columns (omitted printing of 307 columns)</p><table class="data-frame"><thead><tr><th></th><th>features</th><th>feature_intelligence1</th><th>feature_intelligence2</th><th>feature_intelligence3</th></tr><tr><th></th><th title="String">String</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>feature_intelligence1</td><td>1.0</td><td>-0.0141565</td><td>-0.0244041</td></tr><tr><th>2</th><td>feature_intelligence2</td><td>-0.0141565</td><td>1.0</td><td>0.905315</td></tr><tr><th>3</th><td>feature_intelligence3</td><td>-0.0244041</td><td>0.905315</td><td>1.0</td></tr><tr><th>4</th><td>feature_intelligence4</td><td>0.652596</td><td>-0.0280969</td><td>-0.0410859</td></tr><tr><th>5</th><td>feature_intelligence5</td><td>0.0698683</td><td>0.184372</td><td>0.17387</td></tr></tbody></table></div>




```julia
first(stack(feature_corrs), 5)
```




<div class="data-frame"><p>5 rows × 3 columns</p><table class="data-frame"><thead><tr><th></th><th>features</th><th>variable</th><th>value</th></tr><tr><th></th><th title="String">String</th><th title="String">String</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>feature_intelligence1</td><td>feature_intelligence1</td><td>1.0</td></tr><tr><th>2</th><td>feature_intelligence2</td><td>feature_intelligence1</td><td>-0.0141565</td></tr><tr><th>3</th><td>feature_intelligence3</td><td>feature_intelligence1</td><td>-0.0244041</td></tr><tr><th>4</th><td>feature_intelligence4</td><td>feature_intelligence1</td><td>0.652596</td></tr><tr><th>5</th><td>feature_intelligence5</td><td>feature_intelligence1</td><td>0.0698683</td></tr></tbody></table></div>




```julia
tdf = stack(feature_corrs)
tdf = tdf[coalesce.(tdf.variable .< tdf.features, false), :]

sort!(tdf, :value)
vcat(first(tdf, 5), last(tdf, 5))
```




<div class="data-frame"><p>10 rows × 3 columns</p><table class="data-frame"><thead><tr><th></th><th>features</th><th>variable</th><th>value</th></tr><tr><th></th><th title="String">String</th><th title="String">String</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>feature_constitution9</td><td>feature_constitution112</td><td>-0.855008</td></tr><tr><th>2</th><td>feature_constitution46</td><td>feature_constitution33</td><td>-0.83031</td></tr><tr><th>3</th><td>feature_constitution60</td><td>feature_constitution112</td><td>-0.820694</td></tr><tr><th>4</th><td>feature_constitution87</td><td>feature_constitution46</td><td>-0.815888</td></tr><tr><th>5</th><td>feature_constitution33</td><td>feature_constitution112</td><td>-0.759084</td></tr><tr><th>6</th><td>feature_constitution7</td><td>feature_constitution27</td><td>0.94892</td></tr><tr><th>7</th><td>feature_constitution79</td><td>feature_constitution13</td><td>0.949139</td></tr><tr><th>8</th><td>feature_wisdom39</td><td>feature_wisdom31</td><td>0.954984</td></tr><tr><th>9</th><td>feature_wisdom7</td><td>feature_wisdom46</td><td>0.963706</td></tr><tr><th>10</th><td>feature_wisdom2</td><td>feature_wisdom12</td><td>0.968062</td></tr></tbody></table></div>



### The correlation can change over time
You can see this by comparing feature correlations on the first half and second half on the training set


```julia
df₁ = df[coalesce.(eras .<= median(eras), false), :]
df₂ = df[coalesce.(eras .> median(eras), false), :]

corr₁ = DataFrame(cor(Matrix(df₁[!, names(df₁, features)])), features)
insertcols!(corr₁, 1, :features => features)
corr₁ = stack(corr₁)
corr₁ = corr₁[coalesce.(corr₁.variable .< corr₁.features, false), :]

corr₂ = DataFrame(cor(Matrix(df₂[!, names(df₂, features)])), features)
insertcols!(corr₂, 1, :features => features)
corr₂ = stack(corr₂)
corr₂ = corr₂[coalesce.(corr₂.variable .< corr₂.features, false), :]

tdf = leftjoin(corr₁, corr₂, on = [:variable, :features], makeunique=true)
rename!(tdf, [:value, :value_1] .=> [:corr₁, :corr₂])
tdf.corr_diff = tdf.corr₂ - tdf.corr₁
sort!(tdf, :corr_diff)

vcat(first(tdf,5), last(tdf,5))
```




<div class="data-frame"><p>10 rows × 5 columns</p><table class="data-frame"><thead><tr><th></th><th>features</th><th>variable</th><th>corr₁</th><th>corr₂</th><th>corr_diff</th></tr><tr><th></th><th title="String">String</th><th title="String">String</th><th title="Float64">Float64</th><th title="Union{Missing, Float64}">Float64?</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>feature_intelligence9</td><td>feature_intelligence11</td><td>0.0913519</td><td>-0.128851</td><td>-0.220203</td></tr><tr><th>2</th><td>feature_intelligence10</td><td>feature_dexterity12</td><td>0.548931</td><td>0.343117</td><td>-0.205814</td></tr><tr><th>3</th><td>feature_intelligence11</td><td>feature_dexterity9</td><td>0.0787148</td><td>-0.12707</td><td>-0.205785</td></tr><tr><th>4</th><td>feature_dexterity12</td><td>feature_dexterity1</td><td>0.653528</td><td>0.447942</td><td>-0.205587</td></tr><tr><th>5</th><td>feature_intelligence11</td><td>feature_intelligence10</td><td>0.0750222</td><td>-0.130511</td><td>-0.205534</td></tr><tr><th>6</th><td>feature_wisdom22</td><td>feature_intelligence8</td><td>-0.0883461</td><td>0.117772</td><td>0.206119</td></tr><tr><th>7</th><td>feature_wisdom43</td><td>feature_intelligence4</td><td>-0.102438</td><td>0.103758</td><td>0.206197</td></tr><tr><th>8</th><td>feature_wisdom33</td><td>feature_intelligence4</td><td>-0.0789296</td><td>0.133664</td><td>0.212593</td></tr><tr><th>9</th><td>feature_wisdom43</td><td>feature_intelligence8</td><td>-0.121306</td><td>0.115194</td><td>0.236501</td></tr><tr><th>10</th><td>feature_wisdom33</td><td>feature_intelligence8</td><td>-0.0917593</td><td>0.150549</td><td>0.242308</td></tr></tbody></table></div>



## Some features are predictive on their own


```julia
feature_scores = 
    Dict(feature => numerai_score(df.target, df[!, feature], df) 
    for feature in features);
```


```julia
sort(collect(feature_scores), by=x->x[2])
```




    310-element Vector{Pair{String, Float64}}:
          "feature_dexterity7" => -0.011504914975281387
          "feature_dexterity6" => -0.011161569760516648
          "feature_dexterity4" => -0.011051275935746707
          "feature_charisma69" => -0.010221311263804505
         "feature_dexterity11" => -0.010198611285978189
           "feature_charisma9" => -0.01005025481510148
         "feature_dexterity12" => -0.008647990681418269
         "feature_dexterity14" => -0.008611934226827449
          "feature_dexterity3" => -0.007607475423078082
      "feature_constitution91" => -0.007343474478571397
          "feature_dexterity8" => -0.0072798010318430835
      "feature_constitution56" => -0.007206474137472462
     "feature_constitution110" => -0.007154545491821277
                               ⋮
           "feature_strength4" => 0.010184232051437234
          "feature_charisma81" => 0.010224703123621929
          "feature_charisma46" => 0.01039946229179084
          "feature_charisma66" => 0.010507207995266913
          "feature_charisma54" => 0.010507614931830764
           "feature_charisma6" => 0.010580151207446348
          "feature_charisma76" => 0.010608014161745269
          "feature_charisma18" => 0.010697616377184683
          "feature_charisma19" => 0.01071636820945142
          "feature_charisma37" => 0.01089177370157534
          "feature_strength14" => 0.01160884774563975
          "feature_strength34" => 0.012487781133717898




```julia
# Single features do not work consistently though

by_era_correlation = 
    sort(Dict(values(erano)[1] => cor(tdf.target, tdf.feature_strength34)
         for (erano, tdf) in pairs(groupby(df, :erano))))
    
plot(by_era_correlation)
```




    
![svg](_posts/mdimages/analysis_and_tips_julia_18_0.svg)
    




```julia
# With a rolling 10 era average you can see some trends

function rolling_mean(arr, n)
    rs = cumsum(arr)[n:end] .- cumsum([0.0; arr])[1:end-n]
    return rs ./ n
end

n_window = 10

plot(Dict(zip(collect(n_window-1:length(by_era_correlation)), 
        rolling_mean(collect(values(by_era_correlation)),n_window))))
```




    
![svg](_posts/mdimages/analysis_and_tips_julia_19_0.svg)
    



# Gotcha: MSE looks worse than correlation out of sample
Models will generally be overconfident, so even if they are good at ranking rows, the Mean-Squared-Error of the residuals could be larger than event the Mean-Squared-Error of the target (r-squared<0)


```julia
df₁ = df[coalesce.(eras .<= median(eras), false), :]
df₂ = df[coalesce.(eras .> median(eras), false), :];
```


```julia
# This is using MLJ, Julia's homegrown machine-learning library

Linear = @load LinearRegressor pkg=MLJLinearModels verbosity=0
linear = Linear()

lin₁ = machine(linear, df₁[!, names(df₁, features)], df₁.target)
MLJ.fit!(lin₁, verbosity=0)

lin₂ = machine(linear, df₂[!, names(df₂, features)], df₂.target)
MLJ.fit!(lin₂, verbosity=0);
```


```julia
# Note in particular that the R-squared of (train_on_1, eval_on_2) is slightly negative!

r2₁ = [
    r2_score(dfₓ.target, MLJ.predict(model, dfₓ[!, names(dfₓ, features)]))
        for dfₓ in [df₁, df₂]
    for model in [lin₁, lin₂]]

DataFrame(reshape(r2₁, 2, 2), ["eval_on_1","eval_on_2"])
```




<div class="data-frame"><p>2 rows × 2 columns</p><table class="data-frame"><thead><tr><th></th><th>eval_on_1</th><th>eval_on_2</th></tr><tr><th></th><th title="Float64">Float64</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>0.00409275</td><td>-0.000543157</td></tr><tr><th>2</th><td>0.000574622</td><td>0.00315522</td></tr></tbody></table></div>




```julia
# Note in particular that the correlation of (train_on_1, eval_on_2) is quite decent (comparatively)
corrs = [
    numerai_score(MLJ.predict(model, dfₓ[!, names(dfₓ, features)]), dfₓ.target, dfₓ)
        for dfₓ in [df₁, df₂]
    for model in [lin₁, lin₂]]

DataFrame(reshape(corrs, 2, 2), ["eval_on_1","eval_on_2"])
```




<div class="data-frame"><p>2 rows × 2 columns</p><table class="data-frame"><thead><tr><th></th><th>eval_on_1</th><th>eval_on_2</th></tr><tr><th></th><th title="Float64">Float64</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>0.058282</td><td>0.0287217</td></tr><tr><th>2</th><td>0.0319412</td><td>0.0528229</td></tr></tbody></table></div>




```julia
# This can be be run with XGB as well

XGB = @load XGBoostRegressor pkg=XGBoost verbosity=0
xgb = XGB()

xgb₁ = machine(xgb, df₁[!, names(df₁, features)], df₁.target)
MLJ.fit!(xgb₁, verbosity=0)

xgb₂ = machine(xgb, df₂[!, names(df₂, features)], df₂.target)
MLJ.fit!(xgb₂, verbosity=0);
```


```julia
r2₂ = [
    r2_score(dfₓ.target, MLJ.predict(model, dfₓ[!, names(dfₓ, features)]))
        for dfₓ in [df₁, df₂]
    for model in [xgb₁, xgb₂]]

DataFrame(reshape(r2₂, 2, 2), ["eval_on_1","eval_on_2"])
```




<div class="data-frame"><p>2 rows × 2 columns</p><table class="data-frame"><thead><tr><th></th><th>eval_on_1</th><th>eval_on_2</th></tr><tr><th></th><th title="Float64">Float64</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>0.123117</td><td>-0.0237936</td></tr><tr><th>2</th><td>-0.0199959</td><td>0.12788</td></tr></tbody></table></div>




```julia
corrs2 = [
    numerai_score(MLJ.predict(model, dfₓ[!, names(dfₓ, features)]), dfₓ.target, dfₓ)
        for dfₓ in [df₁, df₂]
    for model in [xgb₁, xgb₂]]

DataFrame(reshape(corrs2, 2, 2), ["eval_on_1","eval_on_2"])
```




<div class="data-frame"><p>2 rows × 2 columns</p><table class="data-frame"><thead><tr><th></th><th>eval_on_1</th><th>eval_on_2</th></tr><tr><th></th><th title="Float64">Float64</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>0.383874</td><td>0.0240443</td></tr><tr><th>2</th><td>0.0229365</td><td>0.392287</td></tr></tbody></table></div>



# Gotcha:  {0, 1} are noticeably different from {0.25, 0.75}
This makes training a classifier one-versus-rest behave counterintuitively.

Specifically, the 0-vs-rest and 1-vs-rest classifiers seem to learn how to pick out extreme targets, and their predictions are the most correlated


```julia
# Mostly doing this in Scikitlearn.JL due to no predict_proba (that I'm aware of) in MLJ

logistic = LogisticRegression()
ScikitLearn.fit!(logistic, Matrix(df[!, names(df, features)]), convert.(Int, df.target*4))
ScikitLearn.score(logistic, Matrix(df[!, names(df, features)]), convert.(Int, df.target*4))
```

    C:\Users\Justin\.julia\conda\3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    0.5012315467270351




```julia
# The first and last class are highly correlated
log_corrs = cor(transpose(ScikitLearn.predict_proba(logistic, Matrix(df[!, names(df, features)]))), dims=2)
display(log_corrs)

heatmap(log_corrs,  c=palette(:RdYlGn))
```


    5×5 Matrix{Float64}:
      1.0        0.468155  -0.903881   0.42197    0.947252
      0.468155   1.0       -0.704718   0.517207   0.428423
     -0.903881  -0.704718   1.0       -0.71843   -0.914418
      0.42197    0.517207  -0.71843    1.0        0.498854
      0.947252   0.428423  -0.914418   0.498854   1.0





    
![svg](_posts/mdimages/analysis_and_tips_julia_30_1.svg)
    




```julia
# In-sample correlation

prob_matrix = ScikitLearn.predict_proba(logistic, Matrix(df[!, names(df, features)]))
classes = logistic.classes_
numerai_score(df.target, prob_matrix * classes, df)
```




    0.050658929537343786




```julia
# A standard linear model has a slightly higher correlation
linear = LinearRegression()
ScikitLearn.fit!(linear, Matrix(df[!, names(df, features)]), df.target)
ScikitLearn.score(linear, Matrix(df[!, names(df, features)]), df.target)
preds = ScikitLearn.predict(linear, Matrix(df[!, names(df, features)]))
numerai_score(df.target, preds, df)
```




    0.05107803901831943



# Gotcha: eras are homogenous, but different from each other
##  Random cross-validation will look much better than cross-validating by era

Even for a simple linear model, taking a random shuffle reports a correlation of 4.3%, but a time series split reports a lower score of 3.4%


```julia
#linear = LinearRegression()
#ScikitLearn.fit!(linear, Matrix(df[!, names(df, features)]), df.target)
```


```julia
crossvalidators = [KFold(5), KFold(5, shuffle = true), GroupKFold(5), TimeSeriesSplit(5)]

for cv in crossvalidators
    println(cv)
    println(
        mean(
            cross_val_score(estimator = LinearRegression(),
                X = Matrix(df[!, names(df, features)]),
                y = df.target,
                cv = cv,
                groups = eras,
                scoring = make_scorer(cor, greater_is_better = true)
            )
        )
    )
end
```

    PyObject KFold(n_splits=5, random_state=None, shuffle=False)
    0.03332624500455265
    PyObject KFold(n_splits=5, random_state=None, shuffle=True)
    0.039196207369748895
    PyObject GroupKFold(n_splits=5)
    0.03475937229926111
    PyObject TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    0.030947709608331396
    

## Eras can be more or less applicable to other eras
You can test this be splitting the eras into blocks of 10, training on each block, and evaluating on each other block.


```julia
eras10 = (eras .÷ 10) * 10
countmap(eras10)
```




    Dict{Int64, Int64} with 13 entries:
      20  => 37444
      110 => 45070
      60  => 46831
      30  => 41101
      0   => 24515
      80  => 43971
      90  => 45609
      40  => 43439
      70  => 40403
      50  => 48186
      10  => 34600
      120 => 4532
      100 => 46107




```julia
gdf = copy(df)
gdf[:, :eras10 ] = eras10
gdf = groupby(filter(row -> row[:eras10] < 120, gdf), :eras10);
results10 = DataFrame(train_era = Int32[], test_era = Int32[], value = Float32[])

for train_era in keys(gdf)
    
    println(train_era[1])
    
    gdf₁ = gdf[train_era]
    model = LinearRegression()
    ScikitLearn.fit!(model, Matrix(gdf₁[!, names(gdf₁, features)]), gdf₁.target)
    
    for test_era in keys(gdf)
        
        gdf₂ = gdf[test_era]
        
        push!(results10, [train_era[1], 
                         test_era[1], 
                         cor(gdf₂.target, ScikitLearn.predict(model, Matrix(gdf₂[!, names(gdf₂, features)])))])
    end
end
```

    0
    10
    20
    30
    40
    50
    60
    70
    80
    90
    100
    110
    


```julia
results_df = unstack(results10, :test_era, :value)
```




<div class="data-frame"><p>12 rows × 13 columns (omitted printing of 6 columns)</p><table class="data-frame"><thead><tr><th></th><th>train_era</th><th>0</th><th>10</th><th>20</th><th>30</th><th>40</th><th>50</th></tr><tr><th></th><th title="Int32">Int32</th><th title="Union{Missing, Float32}">Float32?</th><th title="Union{Missing, Float32}">Float32?</th><th title="Union{Missing, Float32}">Float32?</th><th title="Union{Missing, Float32}">Float32?</th><th title="Union{Missing, Float32}">Float32?</th><th title="Union{Missing, Float32}">Float32?</th></tr></thead><tbody><tr><th>1</th><td>0</td><td>0.14615</td><td>0.0321283</td><td>0.0354025</td><td>0.0287675</td><td>0.0221984</td><td>0.00701235</td></tr><tr><th>2</th><td>10</td><td>0.0421759</td><td>0.114813</td><td>0.0287059</td><td>0.0298504</td><td>0.0336937</td><td>0.00471899</td></tr><tr><th>3</th><td>20</td><td>0.0431496</td><td>0.0334976</td><td>0.113055</td><td>0.0366226</td><td>0.0167489</td><td>0.00565709</td></tr><tr><th>4</th><td>30</td><td>0.0357169</td><td>0.0339307</td><td>0.0396031</td><td>0.109884</td><td>0.0402888</td><td>0.0208269</td></tr><tr><th>5</th><td>40</td><td>0.035735</td><td>0.0417183</td><td>0.0204626</td><td>0.0403498</td><td>0.100257</td><td>0.0144214</td></tr><tr><th>6</th><td>50</td><td>0.015032</td><td>0.00959667</td><td>0.00685722</td><td>0.0242685</td><td>0.0151326</td><td>0.104185</td></tr><tr><th>7</th><td>60</td><td>0.00690366</td><td>0.0159851</td><td>0.00419466</td><td>0.0195585</td><td>0.012405</td><td>0.00967648</td></tr><tr><th>8</th><td>70</td><td>0.034285</td><td>0.0252239</td><td>0.0220385</td><td>0.0285308</td><td>0.0232155</td><td>0.00198308</td></tr><tr><th>9</th><td>80</td><td>0.0395826</td><td>0.0268682</td><td>0.0115186</td><td>0.0217091</td><td>0.0177472</td><td>0.00252007</td></tr><tr><th>10</th><td>90</td><td>0.0328201</td><td>0.029052</td><td>0.0229233</td><td>0.031348</td><td>0.0199844</td><td>0.0100413</td></tr><tr><th>11</th><td>100</td><td>0.0283381</td><td>0.0179835</td><td>0.0217198</td><td>0.00991935</td><td>0.00779132</td><td>0.0120947</td></tr><tr><th>12</th><td>110</td><td>0.00181083</td><td>0.0183579</td><td>0.00941574</td><td>0.0067019</td><td>0.0147348</td><td>0.0164994</td></tr></tbody></table></div>




```julia
heatmap(clamp!(Matrix(select(results_df, Not(:train_era))), -.04, .04),  c=palette(:RdYlGn))
```




    
![svg](_posts/mdimages/analysis_and_tips_julia_40_0.svg)
    



Here is an advanced paper that talks about generalization.
Eras can be thought about in the same way that "distributions" or "environments" are talked about here
https://arxiv.org/pdf/1907.02893.pdf

## Gotcha: Since the signal-to-noise ratio is so low, models can take many more iterations than expected, and have scarily high in-sample performance


```julia
df₁ = df[coalesce.(eras .<= median(eras), false), :]
df₂ = df[coalesce.(eras .> median(eras), false), :];
```


```julia
function our_score(preds, dtrain)
    return "score", cor(get_info(dtrain, "label"), preds)
end

dtrain = DMatrix(Matrix(df₁[!, features]), label=df₁.target)
dtest = DMatrix(Matrix(df₂[!, features]), label=df₂.target)
dall = DMatrix(Matrix(df[!, features]), label=df.target);
```


```julia
# This part I wasn't able to replicate perfectly, XGBoost on Julia seems to(?) lack an evals_result to push the data into
# the source code shows only that it prints to stderr - one could redirect it to an IOBuffer and regex parse it into an
# array but realistically the amount of effort isn't worth it, since one can clearly see the out-of-sample performance 
# differneces purely from the numbers printed

param = Dict(
    "eta" => 0.1,
    "max_depth" => 3,
    "objective" => "reg:squarederror",
    "eval_metric" => "rmse"
)


xgboost(dtrain,
    100,
    param = param, 
    watchlist = [(dtrain, "train"), (dtest, "test")], 
    feval = our_score
)
```

    [1]	train-score:0.034205	test-score:0.013370
    [2]	train-score:0.042116	test-score:0.018210
    [3]	train-score:0.044523	test-score:0.020057
    [4]	train-score:0.046625	test-score:0.020591
    [5]	train-score:0.047456	test-score:0.021075
    [6]	train-score:0.050244	test-score:0.022334
    [7]	train-score:0.053165	test-score:0.023862
    [8]	train-score:0.053749	test-score:0.024308
    [9]	train-score:0.055734	test-score:0.025105
    [10]	train-score:0.056863	test-score:0.025744
    [11]	train-score:0.057717	test-score:0.025711
    [12]	train-score:0.058456	test-score:0.026579
    [13]	train-score:0.059670	test-score:0.027121
    [14]	train-score:0.061333	test-score:0.027169
    [15]	train-score:0.062278	test-score:0.027445
    [16]	train-score:0.063603	test-score:0.028017
    [17]	train-score:0.063934	test-score:0.028256
    [18]	train-score:0.065052	test-score:0.028822
    [19]	train-score:0.066291	test-score:0.029125
    [20]	train-score:0.067134	test-score:0.028689
    [21]	train-score:0.068495	test-score:0.029119
    [22]	train-score:0.069201	test-score:0.029029
    [23]	train-score:0.070425	test-score:0.029236
    [24]	train-score:0.071568	test-score:0.029508
    [25]	train-score:0.072137	test-score:0.029973
    [26]	train-score:0.072909	test-score:0.029859
    [27]	train-score:0.073970	test-score:0.030114
    [28]	train-score:0.074752	test-score:0.030397
    [29]	train-score:0.075204	test-score:0.030447
    [30]	train-score:0.075893	test-score:0.030690
    [31]	train-score:0.076471	test-score:0.030660
    [32]	train-score:0.076987	test-score:0.030473
    [33]	train-score:0.077460	test-score:0.030752
    [34]	train-score:0.077869	test-score:0.030746
    [35]	train-score:0.078555	test-score:0.031091
    [36]	train-score:0.078825	test-score:0.031524
    [37]	train-score:0.079270	test-score:0.031895
    [38]	train-score:0.079848	test-score:0.031853
    [39]	train-score:0.080204	test-score:0.031560
    [40]	train-score:0.080906	test-score:0.031729
    [41]	train-score:0.081367	test-score:0.031682
    [42]	train-score:0.082308	test-score:0.031711
    [43]	train-score:0.082807	test-score:0.031745
    [44]	train-score:0.083513	test-score:0.032172
    [45]	train-score:0.084038	test-score:0.032111
    [46]	train-score:0.084551	test-score:0.032103
    [47]	train-score:0.085371	test-score:0.032024
    [48]	train-score:0.086145	test-score:0.032008
    [49]	train-score:0.086439	test-score:0.031888
    [50]	train-score:0.086804	test-score:0.031782
    [51]	train-score:0.087273	test-score:0.032083
    [52]	train-score:0.087956	test-score:0.032262
    [53]	train-score:0.088026	test-score:0.032434
    [54]	train-score:0.088556	test-score:0.032589
    [55]	train-score:0.088790	test-score:0.032618
    [56]	train-score:0.089321	test-score:0.032612
    [57]	train-score:0.089829	test-score:0.032729
    [58]	train-score:0.090087	test-score:0.032762
    [59]	train-score:0.090309	test-score:0.032838
    [60]	train-score:0.091052	test-score:0.032866
    [61]	train-score:0.091601	test-score:0.032716
    [62]	train-score:0.092032	test-score:0.032689
    [63]	train-score:0.092394	test-score:0.032566
    [64]	train-score:0.092747	test-score:0.032685
    [65]	train-score:0.093236	test-score:0.032907
    [66]	train-score:0.093703	test-score:0.032696
    [67]	train-score:0.094115	test-score:0.032824
    [68]	train-score:0.094518	test-score:0.032690
    [69]	train-score:0.094853	test-score:0.032965
    [70]	train-score:0.095340	test-score:0.032935
    [71]	train-score:0.095867	test-score:0.033131
    [72]	train-score:0.096333	test-score:0.033031
    [73]	train-score:0.096724	test-score:0.032923
    [74]	train-score:0.096926	test-score:0.032929
    [75]	train-score:0.097358	test-score:0.032930
    [76]	train-score:0.097798	test-score:0.033127
    [77]	train-score:0.098150	test-score:0.033046
    [78]	train-score:0.098434	test-score:0.033101
    [79]	train-score:0.098696	test-score:0.033058
    [80]	train-score:0.099147	test-score:0.033266
    [81]	train-score:0.099522	test-score:0.033358
    [82]	train-score:0.099870	test-score:0.033457
    [83]	train-score:0.100308	test-score:0.033465
    [84]	train-score:0.100698	test-score:0.033422
    [85]	train-score:0.101070	test-score:0.033549
    [86]	train-score:0.101323	test-score:0.033512
    [87]	train-score:0.101738	test-score:0.033682
    [88]	train-score:0.101921	test-score:0.033735
    [89]	train-score:0.102211	test-score:0.033773
    [90]	train-score:0.102537	test-score:0.033706
    [91]	train-score:0.102779	test-score:0.033658
    [92]	train-score:0.103363	test-score:0.033927
    [93]	train-score:0.103683	test-score:0.033817
    [94]	train-score:0.104193	test-score:0.033676
    [95]	train-score:0.104675	test-score:0.033660
    [96]	train-score:0.104912	test-score:0.033559
    [97]	train-score:0.105215	test-score:0.033580
    [98]	train-score:0.105604	test-score:0.033538
    [99]	train-score:0.105853	test-score:0.033650
    [100]	train-score:0.106177	test-score:0.033704
    




    Booster(Ptr{Nothing} @0x000000018aa6a970)



# The results are sensitive to the choice of parameters, which should be picked through cross-validation


```julia
df₁ = df[coalesce.(eras .<= median(eras), false), :]
df₂ = df[coalesce.(eras .> median(eras), false), :];
```


```julia
XGB = @load XGBoostRegressor pkg=XGBoost verbosity=0
Linear = @load LinearRegressor pkg=MLJLinearModels verbosity=0
Elastic = @load ElasticNetRegressor pkg=MLJLinearModels verbosity=0
```




    ElasticNetRegressor




```julia
models = vcat(
    [Linear()],
    [Elastic(lambda = λ) for λ in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]],
    [XGB()],
    [XGB(eta = 0.01, num_round=1000)],
    [XGB(eta = 0.01, colsample_bytree=0.1, num_round=1000)],
    [XGB(eta = 0.01, colsample_bytree=0.1, num_round=1000, max_depth=5)],
    [XGB(eta = 0.001, colsample_bytree=0.1, num_round=1000, max_depth=5)]
);
```


```julia
for model in models
    print(" -- ", model, "\n")
    mach = machine(model, df₁[!, features], df₁.target)
    MLJ.fit!(mach, verbosity=0)
    outsample = numerai_score(df₂.target, MLJ.predict(mach, df₂[!, features]), df₂)
    insample = numerai_score(df₁.target, MLJ.predict(mach, df₁[!, features]), df₁)
    print("outsample: $outsample, insample: $insample", "\n")
end
```

     -- [34mLinearRegressor @423[39m
    outsample: 0.028025599207339144, insample: 0.06275168899240204
     -- [34mElasticNetRegressor @297[39m
    

    ┌ Warning: Proximal GD did not converge in 1000 iterations.
    └ @ MLJLinearModels C:\Users\Justin\.julia\packages\MLJLinearModels\KE4EE\src\fit\proxgrad.jl:64
    

    outsample: 0.027651106932304964, insample: 0.061801048380003165
     -- [34mElasticNetRegressor @124[39m
    

    ┌ Warning: Proximal GD did not converge in 1000 iterations.
    └ @ MLJLinearModels C:\Users\Justin\.julia\packages\MLJLinearModels\KE4EE\src\fit\proxgrad.jl:64
    

    outsample: 0.027651115862462477, insample: 0.061800987594603896
     -- [34mElasticNetRegressor @012[39m
    

    ┌ Warning: Proximal GD did not converge in 1000 iterations.
    └ @ MLJLinearModels C:\Users\Justin\.julia\packages\MLJLinearModels\KE4EE\src\fit\proxgrad.jl:64
    

    outsample: 0.02765108322778513, insample: 0.06180104261606859
     -- [34mElasticNetRegressor @994[39m
    

    ┌ Warning: Proximal GD did not converge in 1000 iterations.
    └ @ MLJLinearModels C:\Users\Justin\.julia\packages\MLJLinearModels\KE4EE\src\fit\proxgrad.jl:64
    

    outsample: 0.027651162275441423, insample: 0.061801053881855784
     -- [34mElasticNetRegressor @306[39m
    

    ┌ Warning: Proximal GD did not converge in 1000 iterations.
    └ @ MLJLinearModels C:\Users\Justin\.julia\packages\MLJLinearModels\KE4EE\src\fit\proxgrad.jl:64
    

    outsample: 0.027651155706182137, insample: 0.06180105710588575
     -- [34mElasticNetRegressor @576[39m
    

    ┌ Warning: Proximal GD did not converge in 1000 iterations.
    └ @ MLJLinearModels C:\Users\Justin\.julia\packages\MLJLinearModels\KE4EE\src\fit\proxgrad.jl:64
    

    outsample: 0.027651162860262833, insample: 0.06180104657865728
     -- [34mElasticNetRegressor @704[39m
    

    ┌ Warning: Proximal GD did not converge in 1000 iterations.
    └ @ MLJLinearModels C:\Users\Justin\.julia\packages\MLJLinearModels\KE4EE\src\fit\proxgrad.jl:64
    

    outsample: 0.0276511455748371, insample: 0.061801044145696406
     -- [34mElasticNetRegressor @652[39m
    

    ┌ Warning: Proximal GD did not converge in 1000 iterations.
    └ @ MLJLinearModels C:\Users\Justin\.julia\packages\MLJLinearModels\KE4EE\src\fit\proxgrad.jl:64
    

    outsample: 0.02765113464601516, insample: 0.061801044145696406
     -- [34mElasticNetRegressor @867[39m
    

    ┌ Warning: Proximal GD did not converge in 1000 iterations.
    └ @ MLJLinearModels C:\Users\Justin\.julia\packages\MLJLinearModels\KE4EE\src\fit\proxgrad.jl:64
    

    outsample: 0.027651141667503418, insample: 0.06180103987617271
     -- [34mElasticNetRegressor @584[39m
    

    ┌ Warning: Proximal GD did not converge in 1000 iterations.
    └ @ MLJLinearModels C:\Users\Justin\.julia\packages\MLJLinearModels\KE4EE\src\fit\proxgrad.jl:64
    

    outsample: 0.027651141667503418, insample: 0.06180103987617271
     -- [34mXGBoostRegressor @531[39m
    outsample: 0.024815763345799297, insample: 0.4033048140864821
     -- [34mXGBoostRegressor @088[39m
    outsample: 0.03659074980295988, insample: 0.34474251174858644
     -- [34mXGBoostRegressor @046[39m
    outsample: 0.03897987810857149, insample: 0.3118486054318112
     -- [34mXGBoostRegressor @391[39m
    outsample: 0.039683297779925394, insample: 0.20312468355680283
     -- [34mXGBoostRegressor @146[39m
    outsample: 0.034509192497652864, insample: 0.11544644084833998
    

## Gotcha: Models with large exposures to individual features tend to perform poorly or inconsistently out of sample ## 


```julia
# MLJ matches the XGBoost implementation in Python, where num_round == n_estimators 

XGB = @load XGBoostRegressor pkg=XGBoost verbosity=0
xgb = XGB(eta = 0.01, max_depth=5, num_round=1000);
mach = machine(xgb, df₁[!, features], df₁.target)
MLJ.fit!(mach, verbosity=0)

xgb_preds = MLJ.predict(mach, df₂[!, features]);
```


```julia
xgb_preds
```




    248653-element Vector{Float32}:
     0.5092171
     0.51353323
     0.5298325
     0.50988734
     0.50764424
     0.50148475
     0.504006
     0.49748185
     0.49696985
     0.48918203
     0.50696886
     0.51324135
     0.48414978
     ⋮
     0.48918062
     0.47421244
     0.5090075
     0.48367783
     0.47870287
     0.5039986
     0.4987926
     0.49181792
     0.51567954
     0.5039868
     0.48160774
     0.48305735



### Our predictions have correlation > 0.2 in either direction for some single features!
Sure hope those features continue to act as they have in the past!


```julia
cor_list = []
for feature in features
    append!(cor_list, cor(df₂[!, feature], xgb_preds))
end
    
describe(DataFrame(cor_list = cor_list), :all)
```




<div class="data-frame"><p>1 rows × 13 columns (omitted printing of 5 columns)</p><table class="data-frame"><thead><tr><th></th><th>variable</th><th>mean</th><th>std</th><th>min</th><th>q25</th><th>median</th><th>q75</th><th>max</th></tr><tr><th></th><th title="Symbol">Symbol</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>cor_list</td><td>0.0484105</td><td>0.0816448</td><td>-0.229774</td><td>0.00417808</td><td>0.0457962</td><td>0.108646</td><td>0.232515</td></tr></tbody></table></div>




```julia
# treating as one function since Julia gets snippy about subsetting with [!, column] in groupbys

function norm_neut(df, columns, feats, proportion=1.0)
    scores = quantile(Normal(0.0,1.0),(ordinalrank(df[!, columns]) .- 0.5) ./ length(df[!, columns]))
    exposures = Matrix(df[!, feats])
    neutralized = scores - proportion * exposures * (pinv(exposures) * scores)
    return neutralized / std(neutralized)
end;
```


```julia
df₂.preds = xgb_preds

df₂[:, :preds_neutralized] = combine(x -> norm_neut(x, :preds, features, 0.5), groupby(df₂, :erano)).x1

x_min = minimum(df₂.preds_neutralized)
x_max = maximum(df₂.preds_neutralized)
X_std = (df₂.preds_neutralized .- x_min) / (x_max .- x_min)
df₂[!, :preds_neutralized] = X_scaled = X_std * (1 - 0) .+ 0;
```


```julia
describe(df₂.preds_neutralized)
```

    Summary Stats:
    Length:         248653
    Missing Count:  0
    Mean:           0.512301
    Minimum:        0.000000
    1st Quartile:   0.445243
    Median:         0.510633
    3rd Quartile:   0.577324
    Maximum:        1.000000
    Type:           Float64
    

### Now our single feature exposures are much smaller


```julia
cor_list2 = []
for feature in features
    append!(cor_list2, cor(df₂[!, feature], df₂.preds_neutralized))
end
    
describe(DataFrame(cor_list2 = cor_list2), :all)
```




<div class="data-frame"><p>1 rows × 13 columns (omitted printing of 5 columns)</p><table class="data-frame"><thead><tr><th></th><th>variable</th><th>mean</th><th>std</th><th>min</th><th>q25</th><th>median</th><th>q75</th><th>max</th></tr><tr><th></th><th title="Symbol">Symbol</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th><th title="Float64">Float64</th></tr></thead><tbody><tr><th>1</th><td>cor_list2</td><td>0.0361127</td><td>0.0531799</td><td>-0.146887</td><td>0.00775735</td><td>0.0337416</td><td>0.0753398</td><td>0.155376</td></tr></tbody></table></div>



### Our overall score goes down, but the scores are more consistent than before. This leads to a higher sharpe


```julia
unbalanced_scores_per_era = combine(x -> cor(x.preds, x.target), groupby(df₂, :era))
balanced_scores_per_era = combine(x -> cor(x.preds_neutralized, x.target), groupby(df₂, :era));
```


```julia
println("score for high feature exposure: ", mean(unbalanced_scores_per_era.x1))
println("score for balanced feature expo: ", mean(balanced_scores_per_era.x1))

println("std for high feature exposure: ", std(unbalanced_scores_per_era.x1))
println("std for balanced feature expo: ", std(balanced_scores_per_era.x1))

println("sharpe for high feature exposure: ", mean(unbalanced_scores_per_era.x1)/std(unbalanced_scores_per_era.x1))
println("sharpe for balanced feature expo: ", mean(balanced_scores_per_era.x1)/std(balanced_scores_per_era.x1))
```

    score for high feature exposure: 0.0368068530006
    score for balanced feature expo: 0.03288299396994343
    std for high feature exposure: 0.03848940885417861
    std for balanced feature expo: 0.031585501268560856
    sharpe for high feature exposure: 0.9562852248535912
    sharpe for balanced feature expo: 1.0410787433876838
    


```julia
describe(balanced_scores_per_era.x1)
```

    Summary Stats:
    Length:         56
    Missing Count:  0
    Mean:           0.032883
    Minimum:        -0.065154
    1st Quartile:   0.013038
    Median:         0.030797
    3rd Quartile:   0.061884
    Maximum:        0.091098
    Type:           Float64
    


```julia
describe(unbalanced_scores_per_era.x1)
```

    Summary Stats:
    Length:         56
    Missing Count:  0
    Mean:           0.036807
    Minimum:        -0.085174
    1st Quartile:   0.014656
    Median:         0.033550
    3rd Quartile:   0.062175
    Maximum:        0.112687
    Type:           Float64
    
