import math 

def pearson_coefficient (user1_rating,user2_rating):
    average_user1= sum(user1_rating)/len(user1_rating)
    average_user2= sum(user2_rating)/len(user2_rating) 
    numerator = 0 
    denominator_part1= 0  
    denominator_part2= 0  
    for index,_ in enumerate(user2_rating):
        numerator+=(user1_rating[index] - average_user1)*(user2_rating[index] - average_user2)
        denominator_part1+=(user1_rating[index] - average_user1)**2 
        denominator_part2+=(user2_rating[index] - average_user2)**2 
    if denominator_part1 == 0 or denominator_part2 == 0: 
        return 0.0
    return  float(numerator) / ( math.sqrt(denominator_part1) * math.sqrt(denominator_part2))

def raw_cosine(user_rating, active_user_rating):
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


    

    
