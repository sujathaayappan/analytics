
# Load the libraries
library(arules)
library(arulesViz)
library(datasets)
library(readr)


#vlookup in R

prod_tran <- read.csv("C:\\Users\\admin\\Desktop\\marketbasketanalysis\\product order1.csv")
                      
prod_mast <- read.csv("C:\\Users\\admin\\Desktop\\marketbasketanalysis\\products.csv")
                      
Trans1 <- merge(prod_tran, prod_mast, by.x = "product_id", by.y = "product_id", all.x = TRUE)
View(Trans1)

aisles <- read.csv("C:\\Users\\admin\\Desktop\\marketbasketanalysis\\aisles.csv")

Trans1 <- merge(Trans1, aisles, by.x = "aisle_id", by.y = "aisle_id", all.x = TRUE)

dept <- read.csv("C:\\Users\\admin\\Desktop\\marketbasketanalysis\\departments.csv")

Trans1 <- merge(Trans1, dept, by.x = "department_id", by.y = "department_id", all.x = TRUE)

order <- read.csv("C:\\Users\\admin\\Desktop\\marketbasketanalysis\\order1.csv")
head(order)

Trans1 <- merge(Trans1, order, by.x = "order_id", by.y = "order_id", all.x = TRUE)
head(Trans1)
View(Trans1)

summary(Trans1)

Trans2 <- na.omit(Trans1)
AllTrans<- lapply(Trans2, factor)

Tran_rule <- apriori(data = AllTrans)

Tran_rule <- apriori(AllTrans,parameter=list(support=0.01,confidence=0.5))


# Create an item frequency plot for the top 20 items
table(Trans2$aisle)
barplot(table(Trans2$aisle))


str(Trans2)
Trans3 <- read.transactions("Product order1.csv", format = "single", sep = ",", 
                            cols = c("order_id", "department_id"))



Trans3 <- discretize(Trans2$user_id)
Trans3 <- discretize(Trans2$order_number)
Trans3 <- discretize(Trans2$order_dow)
Trans3 <- discretize(Trans2$days_since_prior_order)

Tran_rule <- apriori(Trans3)
head(Trans3)
summary(Trans3)
Trans3 <- read.transactions("Product order1.csv", format = "single", sep = ",", 
                            cols = c("order_id", "department_id"))
