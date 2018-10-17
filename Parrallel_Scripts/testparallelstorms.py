import pstromsinabox
x1, x2 = [345, 375]
y1, y2 = [10, 18]
idstring = 'ptest'
size_of_storm = 5000
c = pstromsinabox.storminbox(x1, x2, y1, y2, size_of_storm, idstring)
c.genstormboxcsv()
