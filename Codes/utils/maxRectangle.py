#%%
class maxRectangle():
    def maxHist(self, row):
        # Create an empty stack. The stack holds indexes of hist array / The bars stored
        # in stack are always in increasing order of their heights.
        stack = []
 
        top_val = 0
        max_area = 0
        area = 0
        iniy, endy, height = 0, 0, 0
 
        # Run through all bars of given histogram (or row)
        i = 0
        while (i < len(row)):
 
            # If this bar is higher than the bar on top stack, push it to stack
            if (len(stack) == 0) or (row[stack[-1]] <= row[i]):
                stack.append(i)
                i += 1
            else:
 
                # If this bar is lower than top of stack, then calculate area of rectangle with
                # stack top as the smallest (or minimum height) bar.
                top_val = row[stack.pop()]
                if top_val == 0:
                    stack = []
                    continue

                if len(stack):
                    area = top_val * (i - stack[-1] - 1)
                else:
                    area = top_val * i

                if area > max_area:
                    max_area = area
                    iniy = stack[-1] + 1 if len(stack) else 0
                    endy = i
                    height = top_val

        # Now pop the remaining bars from stack and calculate area with every popped
        # bar as the smallest bar
        while (len(stack)):
            top_val = row[stack.pop()]
            if top_val == 0:
                stack = []
                continue

            if len(stack):
                area = top_val * (i - stack[-1] - 1)
            else:
                area = top_val * i
 
            if area > max_area:
                max_area = area
                iniy = stack[-1] + 1 if len(stack) else 0
                endy = i
                height = top_val
 
        return max_area, iniy, endy, height
 
    # Returns area of the largest rectangle  with all 1s in A
    # it also returns the left corner coordinates with the correspondent size:
    # [x, y, size_x, size_y]
    def Calc(self, A):
 
        # Calculate area for first row and
        # initialize it as result
        result, iniy, endy, height = self.maxHist(A[0])
        idx = [0, iniy, height, endy-iniy]
 
        # iterate over row to find maximum rectangular
        # area considering each row as histogram
        for i in range(1, len(A)):
 
            for j in range(len(A[i])):
 
                # if A[i][j] is 1 then add A[i -1][j]
                if (A[i][j]):
                    A[i][j] += A[i - 1][j]

            # Update result if area with current
            # row (as last row) of rectangle) is more
            area, iniy, endy, height = self.maxHist(A[i])

            if area > result:
                result = area
                idx = [i-height+1, iniy, height, endy-iniy]
 
        return result, idx
 
 
# Driver Code
if __name__ == '__main__':
    # A = [[0, 1, 1, 0],
    #      [1, 1, 1, 1],
    #      [1, 1, 1, 1],
    #      [1, 1, 0, 0]]

    A=    [
            [1,1,1,1,1,0,0,0],
            [1,1,1,1,1,0,0,0],
            [1,1,1,1,1,0,0,0],
            [1,1,1,0,0,1,1,1],
            [1,1,1,0,0,1,1,1],
            [1,1,1,0,0,1,1,1],
            [0,0,0,0,0,0,0,0]
        ]
    

    ans = maxRectangle()
 
    print("Area of maximum rectangle is",
          ans.Calc(A))
 
# This code is contributed
# by Aaryaman Sharma