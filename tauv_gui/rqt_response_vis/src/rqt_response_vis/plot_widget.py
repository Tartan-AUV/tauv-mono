#!/usr/bin/env python

# Copyright (c) 2011, Dorian Scholz, TU Darmstadt
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of the TU Darmstadt nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
import rospkg
import roslib

from python_qt_binding import loadUi
from python_qt_binding.QtCore import Qt, QTimer, qWarning, Slot
from python_qt_binding.QtGui import QIcon
from python_qt_binding.QtWidgets import QAction, QMenu, QWidget

import rospy

from rqt_py_common.topic_completer import TopicCompleter
from rqt_py_common import topic_helpers

# UNUSED: Just here as an example!

class PlotWidget(QWidget):
    _redraw_interval = 40

    def __init__(self, initial_topics=None, start_paused=False):
        super(PlotWidget, self).__init__()
        self.setObjectName('PlotWidget')

        rp = rospkg.RosPack()
        ui_file = os.path.join(rp.get_path('rqt_plot'), 'resource', 'plot.ui')
        loadUi(ui_file, self.ui)
        self.data_plot = None

        # init and start update timer for plot
        self._update_plot_timer = QTimer(self)
        self._update_plot_timer.timeout.connect(self.update_plot)
    #
    # def switch_data_plot_widget(self, data_plot):
    #     self.enable_timer(enabled=False)
    #
    #     self.data_plot_layout.removeWidget(self.data_plot)
    #     if self.data_plot is not None:
    #         self.data_plot.close()
    #
    #     self.data_plot = data_plot
    #     self.data_plot_layout.addWidget(self.data_plot)
    #     self.data_plot.autoscroll(self.autoscroll_checkbox.isChecked())
    #
    #     if self._initial_topics:
    #         for topic_name in self._initial_topics:
    #             self.add_topic(topic_name)
    #         self._initial_topics = None
    #     else:
    #         for topic_name, rosdata in self._rosdata.items():
    #             data_x, data_y = rosdata.next()
    #             self.data_plot.add_curve(topic_name, topic_name, data_x, data_y)
    #
    #     self._subscribed_topics_changed()

    @Slot(bool)
    def on_pause_button_clicked(self, checked):
        self.enable_timer(not checked)

    @Slot(bool)
    def on_autoscroll_checkbox_clicked(self, checked):
        self.data_plot.autoscroll(checked)
        if checked:
            self.data_plot.redraw()

    @Slot()
    def on_clear_button_clicked(self):
        self.clear_plot()

    def update_plot(self):
        if self.data_plot is not None:
            needs_redraw = False
            # for topic_name, rosdata in self._rosdata.items():
            #     try:
            #         data_x, data_y = rosdata.next()
            #         if data_x or data_y:
            #             self.data_plot.update_values(topic_name, data_x, data_y)
            #             needs_redraw = True
            #     except RosPlotException as e:
            #         qWarning('PlotWidget.update_plot(): error in rosplot: %s' % e)
            # if needs_redraw:
            #     self.data_plot.redraw()

    def clear_plot(self):
        for topic_name, _ in self._rosdata.items():
            self.data_plot.clear_values(topic_name)
        self.data_plot.redraw()

    def enable_timer(self, enabled=True):
        if enabled:
            self._update_plot_timer.start(self._redraw_interval)
        else:
            self._update_plot_timer.stop()
