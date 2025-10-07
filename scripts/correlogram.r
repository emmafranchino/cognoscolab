install.packages("GGally")
install.packages("ggplot2")  # Needed for plotting

# Load packages
library(GGally)
library(ggplot2)

data_3_5 <- read.csv("features_total_gpt_3_5.csv")
data_4o <- read.csv("features_total_gpt_4o.csv")
data_2024_humans <- read.csv("features_total_2024_humans.csv")
data_2023_humans <- read.csv("features_total_2023_humans.csv")

# to match the order in the preprint
numeric_data <- data_2024_humans[c("degree_matematica",
"degree_ansia",
"clo_cen_matematica",
"clo_cen_ansia",
"valence_matematica",
"valence_ansia",
"mas_eval",
"mas_social",
"mas_pass_obs",
"mas_total_ma")]

# to match the lables in the preprint
names(numeric_data) <- c("Degree \"math\"",
"Degree \"anxiety\"",
"CC \"math\"",
"CC \"anxiety\"",
"Valence \"math\"",
"Valence \"anxiety\"",
"Evaluation MA",
"Social MA",
"Observation MA",
"Total MA")

# png("correlogram/correlogram_humans_2024.png", width = 1200, height = 800)

# Create the correlogram-like pairs plot
ggpairs(data_2023_humans,
        lower = list(continuous = wrap("smooth", alpha = 0.5, size = 0.5, color = "blue")),
        upper = list(continuous = wrap("cor", size = 5)),
        diag = list(continuous = wrap("barDiag", fill = "blue")))

# dev.off()