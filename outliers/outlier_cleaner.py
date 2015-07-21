#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here

#calculated the error (sum of least squares)
    error = (net_worths-predictions)**2
    #zip up an array
    cleaned_data = zip(ages,net_worths,error)
    #sort by error, lambda x for doing a rowise extraction of error, which is the 1st element of the 3rd element
    cleaned_data = sorted(cleaned_data,key=lambda x:x[2][0])
    #calulate 10% of the points
    number_points_remove = int(0.1*len(net_worths))
    #use all but the last 10% of the points from the sorted array
    cleaned_data = cleaned_data[:-number_points_remove]


    
    return cleaned_data

