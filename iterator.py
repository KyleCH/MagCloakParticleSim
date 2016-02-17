#!/usr/bin/env python3

# Third party modules.
import numpy as np

# Custom modules.
from stepsetter import setstep

def iterator(f, h, method, t, y=np.zeros(6), nmax=1000, exitcond=None,
  trackint=None):
    """Iterates the updating of parameters of a differential equation.
    
    Parameters
    ----------
    f : function
        The differential equation to be solved (approximatly).
    h : float
        The step size, i.e. how much time elapses between each step.
    method : str
        Specifies the method that will be used to approximate the
        solution to f.
    t : float
        The starting time.
    y : numpy.ndarray
        The inital value for the parameters of f other than time.
    nmax : int, optional
        The maximum number of steps to iterate over.
    exitcond : function or None, optional
        The exit condition for the iteration.  If none is given, the
        iteration will terminate after nmax iterations.
    trackint : int or None, optional
        The tracking interval.  The number of steps the iterator should
        calculate between storing the results to the tracking arrays
        (i.e. the iterator will save stor the result of every
        trackint-th step).  If trackint is None, tracking is disabled. 
    """
    
    # Set step function
    step = setstep(f, method, h)
    
    # If tracking is disabled (trackint is None).
    if (trackint is None):
        
        # If there is no exit condition (exitcond is None).
        if exitcond is None:
            for i in range(nmax):
                t, y = step(t, y)
        
        # If there is an exit condition (exitcond is not None).
        else:
            for i in range(nmax):
                t, y = step(t, y)
                if exitcond(t, y):
                    break
        
        return t, y
    
    # If tracking is enabled for all steps.
    elif (trackint == 1):
        
        # Setup tracking arrays.
        t_arr = np.empty(nmax+1)
        y_arr = np.empty((nmax+1, y.size))
        t_arr[0] = t
        y_arr[0, :] = y
        del t, y
        
        # If there is no exit condition (exitcond is None).
        if exitcond is None:
            for i in range(1, nmax+1):
                t_arr[i], y_arr[i, :] = step(t_arr[i-1], y_arr[i-1, :])
        
        # If there is an exit condition (exitcond is not None).
        else:
            for i in range(1, nmax+1):
                # Update time and position.
                t_arr[i], y_arr[i, :] = step(t_arr[i-1], y_arr[i-1, :])
                
                # Terminate loop if exit condition (exitcond) is
                # satisfied and delete excess entries in tracking
                # arrays.
                if exitcond(t_arr[i], y_arr[i, :]):
                    t_arr = np.delete(t_arr, range(i+1, nmax+1), 0)
                    y_arr = np.delete(y_arr, range(i+1, nmax+1), 0)
                    break
        return t_arr, y_arr
    
    # If tracking interval (trackint) is greater than one, takes
    # trackint many steps beetween recording values to tracking arrays.
    elif (trackint > 1):
        
        # Number of steps to track.
        ntrack = 1 + (nmax+trackint-1)//trackint
        
        # Setup tracking arrays.
        t_arr = np.empty(ntrack+1)
        y_arr = np.empty((ntrack+1, y.size))
        t_arr[0] = t
        y_arr[0, :] = y
        
        # If there is no exit condition (exitcond is None).
        if exitcond is None:
            for i in range(1, ntrack):
                for j in range(trackint):
                    t, y = step(t, y)
                t_arr[i], y_arr[i, :] = t, y
            
            # Calculate final tracked values.  This is seperate from
            # main loop to allow for partial intervals when the maximum
            # number of steps (nmax) is not a mutiple of tracking
            # interval (trackint).
            for j in range(nmax - (ntrack-2)*trackint):
                t, y = step(t, y)
            t_arr[i], y_arr[i, :] = t, y
        
        # If there is an exit condition (exitcond is not None).
        else:
            for i in range(1, nmax+1):
                for j in range(trackint):
                    t, y = step(t, y)
                t_arr[i], y_arr[i, :] = t, y
                
                # Terminate loop if exit condition (exitcond) is
                # satisfied and delete excess entries in tracking
                # arrays.
                if exitcond(t_arr[i], y_arr[i, :]):
                    t_arr = np.delete(t_arr, range(i+1, nmax+1), 0)
                    y_arr = np.delete(y_arr, range(i+1, nmax+1), 0)
                    break
            
            # Calculate final tracked values.  This is seperate from
            # main loop to allow for partial intervals when the maximum
            # number of steps (nmax) is not a mutiple of tracking
            # interval (trackint).
            else:
                for j in range(nmax - (ntrack-2)*trackint):
                    t, y = step(t, y)
                t_arr[i], y_arr[i, :] = t, y
        
        return t_arr, y_arr
    
    # Prints error text if tracking interval (trackint) is not valid.
    else:
        print("Error in file \"{}\", function \"iterator\".\n".format(__file__)
            + "    Invalid tracking interval (trackint), trackint must be "
            + "either a positive\n"
            + "    integer or None.  If trackint is None, the track is not "
            + "recorded.")
        raise SystemExit
