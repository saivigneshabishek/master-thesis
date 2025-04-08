

class LifeManagement:
    def __init__(self, cfg, cls, frame_id):
        self.cfg = cfg
        self.age = 1
        self.min_age = self.cfg['min_age'][cls]
        self.max_age = self.cfg['max_age'][cls]
        self.time_since_last_update = 0
        self.cls = cls
        self.time = frame_id
        self.state = 'active' if self.min_age <=1 or frame_id <= self.min_age else 'tentative'

    def predict(self, frame_id):
        self.time_since_last_update += 1
        self.time = frame_id

    def update(self, curr_det):
        if curr_det is not None:
            self.time_since_last_update = 0
            self.age += 1

        if self.state == 'tentative':
            if (self.age >= self.min_age) or (self.time <= self.min_age):
                self.state = 'active'
            elif self.time_since_last_update > 0:
                self.state = 'dead'
            else: pass
        elif self.state == 'active':
            if self.time_since_last_update >= self.max_age:
                self.state = 'dead'
            # if self.time_since_last_update >= self.cfg['tent_age'][self.cls]:
            #     self.state ='tentative'
            #     self.age = 0
            else:pass
        else : 
            raise Exception('Dead tracks should be removed.')