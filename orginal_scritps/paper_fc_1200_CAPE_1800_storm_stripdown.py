if len(fle_u.coord('pressure').points) == 18:
    cube_mslp = fle_mslp[11, :, :].extract(xysmallslice)
    cube_mslp = cube_mslp.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
    q = cube_mslp.data/100.
	cube_T = fle_T[3,:,:].extract(xysmallslice_850_upwards)
	cube_q = fle_q[3,:,:].extract(xysmallslice_850_upwards)
	T_data = cube_T.data
	q_data = cube_q.data

	for p in range(0, T_data.shape[0]):
		for y in range(0, T_data.shape[1]):
			for x in range(0, T_data.shape[2]):
				if T_data[p, y, x] < 100:
					T_data[p, y, x] = float('nan')
					q_data[p, y, x] = float('nan')
	T_collapsed[p] = np.nanmean(T_data[p,:,:])
	q_collapsed[p] = np.nanmean(q_data[p,:,:])
	cube_T = cube_T.collapsed(['latitude','longitude'], iris.analysis.MEAN)
	cube_q = cube_q.collapsed(['latitude','longitude'], iris.analysis.MEAN)
	cube_T.data = T_collapsed
	cube_q.data = q_collapsed
	cube_pressures = cube_q.copy()
	pressures = cube_T[:,:].coord('pressure').points
	pressures = np.ndarray.tolist(pressures)
	pressures.extend([q])
	pressures = np.asarray(pressures)
	fake_T = fle_T[3,:pressure_shape+1,1,1]
	fake_q = fle_q[3,:pressure_shape+1,1,1]
	fake_T.data[:pressure_shape] = T_collapsed
	fake_T.data[pressure_shape] = cube_T15.data
	fake_q.data[:pressure_shape] = q_collapsed
	fake_q.data[pressure_shape] = cube_q15.data
	cube_T = fake_T
	cube_q = fake_q
	cube_pressures = cube_q.copy()
	Ps = np.zeros((len(pressures)),float)
	cube_pressures.data = pressures*100
	if np.sum(all_cube) < 1:
		all_cube = np.zeros((1, len(pressures), 9), float)
	else:
		temp_cube = np.zeros((1, len(pressures), 9), float)

    if len(pressures) == 18:
        for p in range(0, len(pressures)):
            if 710. >= pressures[p] > 690.:
                RH_650hPa.extend([(0.263 * cube_q.data[p] * cube_pressures.data[p])/2.714**((17.67*(cube_T.data[p] - 273.16))/(cube_T.data[p] - 29.65))])
			if all_cube.shape[0] == 1:
				all_cube[0,p,4] = (0.263 * cube_q.data[p] * cube_pressures.data[p])/2.714**((17.67*(cube_T.data[p] - 273.16))/(cube_T.data[p] - 29.65))
				all_cube[0,p,5] = meteocalc.dew_point(temperature = cube_T[p].data - 273.16, humidity = all_cube[0,p,4])
			else:
				temp_cube[0,p,4] = (0.263 * cube_q.data[p] * cube_pressures.data[p])/2.714**((17.67*(cube_T.data[p] - 273.16))/(cube_T.data[p] - 29.65))
				temp_cube[0,p,5] = meteocalc.dew_point(temperature = cube_T[p].data - 273.16, humidity = temp_cube[0,p,4])
			if p < len(pressures)-1:
				if all_cube.shape[0] == 1:
					all_cube[0,p,1] = cube_T[p].data*((cube_mslp.data/cube_pressures.data[p])**(1./5.257) - 1)/0.0065
				else:
					temp_cube[0,p,1] = cube_T[p].data*((cube_mslp.data/cube_pressures.data[p])**(1./5.257) - 1)/0.0065

			else:
				if all_cube.shape[0] == 1:
					all_cube[0,p,1] = 1.5
				else:
					temp_cube[0,p,1] = 1.5

    if all_cube.shape[0] == 1:
        all_cube[0, :, 0] = cube_pressures.data/100
        all_cube[0, :, 2] = cube_T.data
        all_cube[0, :, 3] = cube_q.data
        all_cube[0, :, 6] = p99
        all_cube[0, :, 2] = all_cube[0, :, 2] - 273.16
        all_cube[0, :, 7] = cube_u.data
        all_cube[0, :, 8] = cube_v.data
	    mydata = dict(zip(('hght', 'pres', 'temp', 'dwpt'),
                          (all_cube[0,::-1,1], all_cube[0,::-1,0],
                           all_cube[0,::-1,2], all_cube[0,::-1,5])))
        S = sk.Sounding(soundingdata=mydata)
        parcel = S.get_parcel('mu')
        P_lcl,P_lfc,P_el,CAPE,CIN=S.get_cape(*parcel)
        CAPE_stats[cntr,0] = P_lcl
        CAPE_stats[cntr,1] = P_lfc
        CAPE_stats[cntr,2] = P_el
        CAPE_stats[cntr,3] = CAPE
        CAPE_stats[cntr,4] = CIN
		CAPE_stats[cntr,5] = storms_to_keep[rw,8]
	else:
		temp_cube[0,:,0] = cube_pressures.data/100
		temp_cube[0,:,2] = cube_T.data
		temp_cube[0,:,3] = cube_q.data
		temp_cube[0,:,6] = p99
        temp_cube[0,:,2] = temp_cube[0,:,2] - 273.16
		temp_cube[0,:,7] = cube_u.data
		temp_cube[0,:,8] = cube_v.data
        mydata = dict(zip(('hght','pres','temp','dwpt'),(temp_cube[0,::-1,1],temp_cube[0,::-1,0], temp_cube[0,::-1,2],temp_cube[0,::-1,5])))
        S=sk.Sounding(soundingdata=mydata)
        parcel = S.get_parcel('mu')
        P_lcl,P_lfc,P_el,CAPE,CIN=S.get_cape(*parcel)
        CAPE_stats[cntr,0] = P_lcl
        CAPE_stats[cntr,1] = P_lfc
        CAPE_stats[cntr,2] = P_el
        CAPE_stats[cntr,3] = CAPE
        CAPE_stats[cntr,4] = CIN
		CAPE_stats[cntr,5] = storms_to_keep[rw,8]
		cntr = cntr + 1
	try:
		if temp_cube.shape[1] == all_cube.shape[1]:
			all_cube = np.concatenate((all_cube, temp_cube), axis = 0)
	except UnboundLocalError or ValueError:
		continue
all_cube = np.average(all_cube, axis = 0)
np.savetxt('FC_c2c4_storms_midday_stats_for_TEPHI_1800Z_storms.csv', all_cube, delimiter = ',')
np.savetxt('../csvs/FC_c2c4_1200_CAPE_CIN_1800_STORMS.csv', CAPE_stats, delimiter = ',')
np.savetxt('../csvs/FC_c2c4_1200_700_hPa_RH_1800_storms.csv', RH_650hPa, delimiter = ',')
