for /l %%x in (2700, 100, 5000) do (
    C:\Users\g.werner\AppData\Local\Programs\Python\Python36\python Testbed.py C:\Users\g.werner\eclipse-workspace\GenreDeciderPython\input\train test_results/results_charlstm_%%x.txt %%x 100 charlstm 
)