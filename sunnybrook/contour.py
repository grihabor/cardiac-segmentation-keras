import re

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-.*', ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))

    def __str__(self):
        return '<Contour for case %s, image %d>' % (self.case, self.img_no)

    def __repr__(self):
        return str(self)
    