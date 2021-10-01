def animate_img_function_f_of_time(func, t_iterable, pause_seconds=1, wait_for_a_key_after_stop=True, 
                                  win_title='Animation', width=None, height=None,
                                  repeat_times=1):
    cv2.namedWindow(win_title)
    for n_ in range(repeat_times):
        for t in t_iterable:
            res_img = func(t)

            if width is None and height is not None:
                res_img = resize(res_img, height=height)
            elif width is not None and height is None:
                res_img = resize(res_img, width=width)
            elif width is not None and height is not None:
                res_img = fit_img_center(res_img, width=width, height=height)

            cv2.imshow(win_title, res_img)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyWindow(win_title)
                return
            time.sleep(pause_seconds)
    cv2.destroyWindow(win_title)


