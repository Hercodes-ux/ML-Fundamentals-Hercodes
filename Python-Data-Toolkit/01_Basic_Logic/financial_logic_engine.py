
#Demostrating using OOPS concepts to build a financial logic engine
class InvestAnalyst:
    def __init__(self, principal, rate, years):
        self.principal = float(principal)
        self.rate = float(rate) / 100  # Convert percentage to decimal
        self.years = int(years)
    
    def calcluate_compound_interest(self):  
        #Compund Interest Formula: A = P (1+r/n)^(nt)
        final_amount = self.principal *( 1+ self.rate) ** self.years
        return final_amount





#Execution logic
if __name__ == "__main__":
    print("---Hercodes Professional Investemnt Logic----")
    principal =input("Enter the principal amount: ")
    rate = input("Enter the annual interest rate (in %): ")
    years= input("Enter the number of years: ") 
    investor = InvestAnalyst(principal, rate, years)
    result = investor.calcluate_compound_interest() 
    print(f"Projected amount after {years} years is: {result:.2f}")