library(yaml)
library(feather)
library(lme4)
library(simr)

# Read the configuration path
config <- yaml.load_file("./src/config.yml")

#-----------------------------------------------------------------------
# Read the r dataset
r_dataframe_path <- paste("./", config$r_dataframe, sep = "")
df <- read_feather(r_dataframe_path)
df <- na.omit(df)
df