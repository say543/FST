
windows cmd
// loop example. set paramter exaplmes


CD m_492aef0c-e200-4e91-b8df-94bcd8a75cd9
SET ModelfileSuffix=%1

echo.
echo ======== [Cmd arguments] =========
echo ModelfileSuffix=%ModelfileSuffix%
echo.


FOR /L %%G IN (1,1,2) DO run.cmd ..\i1_febd21f8-0ca4-42b6-8137-2978cf0e5a0d\febd21f8-0ca4-42b6-8137-2978cf0e5a0d ..\i2_17a05a6a-cb86-4883-990a-9fbac88c7495\17a05a6a-cb86-4883-990a-9fbac88c7495 ..\i3_88d1b0e2-218f-4924-aa8e-19ebb179094a\88d1b0e2-218f-4924-aa8e-19ebb179094a ..\o1_e47bb398-4856-409f-8933-1ab89036adca\%%G "assistant_enus_tvs_mv1" 2 3 true base64 "" "false"  

