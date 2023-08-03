import math 

def pearson_coefficient (user_rating,active_user_rating):
    average_user= sum(user_rating)/len(user_rating)
    average_active_user= sum(active_user_rating)/len(active_user_rating) 
    numerator = 0 
    denominator_part1= 0  
    denominator_part2= 0  
    for index,_ in enumerate(active_user_rating):
        numerator+=(user_rating[index] - average_user)*(active_user_rating[index] - average_active_user)
        denominator_part1+=(user_rating[index] - average_user)**2 
        denominator_part2+=(active_user_rating[index] - average_active_user)**2 
    if denominator_part1 == 0 or denominator_part2 == 0: 
        return 0.0
    return  float(numerator) / ( math.sqrt(denominator_part1) * math.sqrt(denominator_part2))

def adjusted_cosine(user_rating, active_user_rating):
    numerator = 0
    denominator_part1 = 0
    denominator_part2 = 0
    for index,_ in enumerate(user_rating):
        numerator += user_rating[index]*active_user_rating[index] 
        denominator_part1+=user_rating[index]**2
        denominator_part2+=active_user_rating[index]**2
    if denominator_part1 == 0 or denominator_part2 == 0:
        return 0.0
    return  float(numerator)/(math.sqrt(float(denominator_part1))*math.sqrt(float(denominator_part2)))


    

    
