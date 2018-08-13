import numpy as np


class Visdom(object):
    """tool for visdom visualization

    """

    def __init__(self, display_id=0, num=3):
        self.idx = display_id
        self.num = num
        self.win = dict()
        if display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=8097)

    def create_vis_line(self, xlabel, ylabel, title, legend, name):
        self.win[name] = self.vis.line(
            X=np.zeros((1, len(legend))),
            Y=np.zeros((1, len(legend))),
            opts=dict(xlabel=xlabel, ylabel=ylabel, title=title, legend=legend)
        )

    def update_vis_line(self, iter, loss, name, update_type, epoch_size=1):
        self.vis.line(
            X=np.ones((1, len(loss))) * iter,
            Y=np.array(loss).reshape((1, len(loss))) / epoch_size,
            win=self.win[name],
            update=update_type
        )
        # initialize
        if iter == 0:
            self.vis.line(
                X=np.zeros((1, len(loss))),
                Y=np.array(loss).reshape((1, len(loss))),
                win=self.win[name],
                update=True
            )

    def create_vis_scatter(self, xlabel, ylabel, name):
        self.win[name] = self.vis.scatter(
            X=np.zeros((1, 2)),
            opts=dict(markersize=10, xlabel=xlabel, ylabel=ylabel))

    # TODO: there is a bug for one points
    def update_vis_scatter(self, x, y, name, first=False):
        self.vis.scatter(
            X=x,
            Y=y,
            win=self.win[name],
            update=['append', 'new'][first]
        )

    def create_vis_image(self, size, title, name):
        self.win[name] = self.vis.image(
            np.random.randn(*size),
            opts=dict(title=title)
        )

    def update_vis_image(self, image, name):
        self.vis.image(image, win=self.win[name])
