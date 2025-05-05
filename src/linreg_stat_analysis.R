# This script uses linear regression model
#   to find out which predictor variables are significant
#   in response to the rent levels in San Francisco.
# The data used in this script should have been preprocessed and cleaned
#   using preprocess_data.py we provided.

data <- read.csv("../data/processed_data.csv")
data <- data[data$market == "San Francisco", ]
head(data)

fit_rent <- function(data) {
  result <- lm(rent ~ . - building_id - market - buiding_name
               - address - region - city - state, data = pre_covid)
  summary(result)
  return(result)
}
# study the correlations before and after covid
pre_covid <- data[data$year < 2020, ]
fit_pre_covid <- fit_rent(pre_covid)

post_covid <- data[data$year >= 2020, ]
fit_post_covid <- fit_rent(post_covid)

# study the trend of rent of a specific building in San Francisco
# can also be the averge rent of all buildings in San Francisco
building <- data[data$building_address == "150 Calidornia St", ]
rent <- ts(building$rent, frequency = 4, start = c(2018, 1))
t <- time(rent)
trend <- coef(lm(rent ~ t))
trend <- trend[1] + trend[2] * t
plot(rent, main = "Rent of 150 Calidornia St in San Francisco")
lines(trend, col = "red")
legend("topleft", legend = c("Rent", "Trend"), col = c("black", "red"), lty = 1)
