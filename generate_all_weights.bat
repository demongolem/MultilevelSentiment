FOR /L %%x IN (1,1,511) DO (echo %%x & C:\Users\g.werner\AppData\Local\Programs\Python\Python36\python CompositeSentiment.py train %%x sm 0 -1 False)