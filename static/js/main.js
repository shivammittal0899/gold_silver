
function addEMA(data) {
    let emaSeries = chart.addSeries(LightweightCharts.LineSeries, {
        color: 'blue',
        lineWidth: 2
    });

    emaSeries.setData(
        data
            .filter(d => d.ema != null)
            .map(d => ({ time: d.time, value: d.ema }))
    );
}

function addVWAP(data) {
    let vwapSeries = chart.addSeries(LightweightCharts.LineSeries, {
        color: 'orange',
        lineWidth: 2
    });

    vwapSeries.setData(
        data
            .filter(d => d.vwap != null)
            .map(d => ({ time: d.time, value: d.vwap }))
    );
}
function addIchimoku(data) {

    const tenkan = chart.addSeries(LightweightCharts.LineSeries, { color: "blue" });
    const kijun = chart.addSeries(LightweightCharts.LineSeries, { color: "red" });

    const spanA = chart.addSeries(LightweightCharts.AreaSeries, {
        topColor: "rgba(0, 200, 0, 0.3)",
        bottomColor: "rgba(0, 200, 0, 0.05)",
        lineColor: "green",
        lineWidth: 1
    });

    const spanB = chart.addSeries(LightweightCharts.AreaSeries, {
        topColor: "rgba(200, 0, 0, 0.3)",
        bottomColor: "rgba(200, 0, 0, 0.05)",
        lineColor: "red",
        lineWidth: 1
    });

    // ✅ NO SHIFT
    tenkan.setData(data.filter(d => d.tenkan != null)
        .map(d => ({ time: d.time, value: d.tenkan })));

    kijun.setData(data.filter(d => d.kijun != null)
        .map(d => ({ time: d.time, value: d.kijun })));

    spanA.setData(data.filter(d => d.spanA != null)
        .map(d => ({ time: d.time, value: d.spanA })));

    spanB.setData(data.filter(d => d.spanB != null)
        .map(d => ({ time: d.time, value: d.spanB })));
}
function addRSI(data) {
    let rsiSeries = chart.addSeries(LightweightCharts.LineSeries, {
        color: 'purple',
        lineWidth: 2,
        priceScaleId: 'rsi-scale'
    });

    chart.priceScale('rsi-scale').applyOptions({
        scaleMargins: {
            top: 0.8,
            bottom: 0
        }
    });

    rsiSeries.setData(
        data
            .filter(d => d.rsi != null)
            .map(d => ({ time: d.time, value: d.rsi }))
    );
}


function getReturnColor(val) {
    if (val === null || val === undefined) return "";

    // Strong negative → dark red
    if (val <= -10) return "background-color:#e60000; color:white; font-weight:bold;";
    
    if (val <= -5) return "color:#cc0000; font-weight:bold;";

    // Mild negative → light red
    if (val < 0) return "color:#ff4d4d; font-weight:bold;";

    // Neutral
    if (val === 0) return "background-color:#f2f2f2; font-weight:bold;";

    // Mild positive → light green
    if (val > 0 && val < 5) return "color:#00cc00; font-weight:bold;";

    // Strong positive → dark green
    if (val >=5 && val < 10) return "color:#004d00; font-weight:bold;";
    
    if (val >= 10) return "background-color:#009933; color:white; font-weight:bold;";

    return "";
}
function getRSIColor(val) {
    if (val === null || val === undefined) return "";

    // Strong negative → dark red
    if (val <= 20) return "background-color:#e60000; color:white; font-weight:bold;";
    
    if (val <= 40) return "color:#cc0000; font-weight:bold;";

    // Neutral
    if (val < 60) return "background-color:#f2f2f2; font-weight:bold;";

    // Strong positive → dark green
    if (val <= 80) return "color:#004d00; font-weight:bold;";
    
    if (val <= 100) return "background-color:#004d00; color:white; font-weight:bold;";

    return "";
}
function getFundamentalColor(val) {
    if (val === null || val === undefined) return "";
    // Strong negative → dark red
    if (val <= 20) return "background-color:#e60000; color:white; font-weight:bold;";
    
    if (val <= 40) return "color:#cc0000; font-weight:bold;";
}

function stock_analysis_tables(data){
    let htmlt = ""
    let htmlhl = ""
    let htmlfr = ""
    let htmlfg = ""
    let htmlft = ""
    console.log("in stock analysis tables")
    data.forEach(d => {
        let sig30 = d.signal_30m || "";
        let sig60 = d.signal_60m || "";
        let sig1d = d.signal_1d || "";

        let color30 =
            sig30.includes("Strong Buy") ? "darkgreen" :
            sig30.includes("Strong Sell") ? "darkred" :
            sig30.includes("Buy") ? "lightgreen" :
            sig30.includes("Sell") ? "red" : "black";
        let color60 =
            sig60.includes("Strong Buy") ? "darkgreen" :
            sig60.includes("Strong Sell") ? "darkred" :
            sig60.includes("Buy") ? "green" :
            sig60.includes("Sell") ? "red" : "black";
        let color1d =
            sig1d.includes("Strong Buy") ? "darkgreen" :
            sig1d.includes("Strong Sell") ? "darkred" :
            sig1d.includes("Buy") ? "green" :
            sig1d.includes("Sell") ? "red" : "black";

        htmlt += `
            <tr>
                <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;
                background:${
                    d.is_fut
                    ? '#fff3cd'
                    : ''
                };

                font-weight:${
                    d.is_fut
                    ? 'bold'
                    : 'normal'
                };">
                    ${d.symbol}
                </td>
                
                <td>${d.industry || '-'}</td>
                <td class="ltp-cell"
                    data-symbol="${d.symbol}"
                    data-ltp="${d.ltp || 0}"
                    style="font-weight:bold;">
                    ${d.ltp?.toFixed(2) || '-'}
                </td>
                <td style="${getReturnColor(d.ret1)}">${d.ret1?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret5)}">${d.ret5?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret15)}">${d.ret15?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret30)}">${d.ret30?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret90)}">${d.ret90?.toFixed(2) || '-'}</td>
                <td style="color:${color30}; font-weight:bold;">${sig30 || '-'}</td>
                <td style="color:${color60}; font-weight:bold;">${sig60 || '-'}</td>
                <td style="color:${color1d}; font-weight:bold;">${sig1d || '-'}</td>
                <td style="${getRSIColor(d.rsi_30m)}">${d.rsi_30m || '-'}</td>
                <td style="${getRSIColor(d.rsi_60m)}">${d.rsi_60m || '-'}</td>
                <td style="${getRSIColor(d.rsi_1d)}">${d.rsi_1d || '-'}</td>
                <td>${d.price_tenkan_30m || '-'}</td>
                <td>${d.price_tenkan_60m || '-'}</td>
                <td>${d.price_tenkan_1d || '-'}</td>
                <td>${d.tenkan_kijun_30m || '-'}</td>
                <td>${d.tenkan_kijun_60m || '-'}</td>
                <td>${d.tenkan_kijun_1d || '-'}</td>
            </tr>
        `;

        htmlhl +=`
            <tr>
                <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;
                background:${
                    d.is_fut
                    ? '#fff3cd'
                    : ''
                };

                font-weight:${
                    d.is_fut
                    ? 'bold'
                    : 'normal'
                };">
                    ${d.symbol}
                </td>
                <td class="ltp-cell"
                    data-symbol="${d.symbol}"
                    data-ltp="${d.ltp || 0}"
                    style="font-weight:bold;">
                    ${d.ltp?.toFixed(2) || '-'}
                </td>
                <td style="${getReturnColor(d.ret1)}">${d.ret1?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retdayHigh)}">${d.retdayHigh?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retdayLow)}">${d.retdayLow?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retweekHigh)}">${d.retweekHigh?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retweekLow)}">${d.retweekLow?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retmonthHigh)}">${d.retmonthHigh?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retmonthLow)}">${d.retmonthLow?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retyearHigh)}">${d.retyearHigh?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retyearLow)}">${d.retyearLow?.toFixed(2) || '-'}</td>
            </tr>
        `;

        htmlfr += `
            <tr>
                <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;">
                    ${d.symbol}
                </td>
                <td style="${getFundamentalColor(d.ret1)}">${d.beta?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.trailingPE?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.forwardPE?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.trailingEPS?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.forwardEPS?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.epsCurrentYear?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.epsForward?.toFixed(2) || '-'}</td>
                
                <td style="${getReturnColor(d.ret1)}">${d.quickRatio?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.currentRatio?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.debtToEquity?.toFixed(2) || '-'}</td>
            </tr>
        `;

        htmlfg += `
            <tr>
                <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;">
                    ${d.symbol}
                </td>
                <td style="${getFundamentalColor(d.ret1)}">${d.earningsQuarterlyGrowth?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.earningsGrowth?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.revenueGrowth?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.revenuePerShare?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.totalCashPerShare?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.profitMargins?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.grossMargins?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.ebitdaMargins?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.enterpriseToRevenue?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.enterpriseToEbitda?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.priceToBook?.toFixed(2) || '-'}</td>
            </tr>
        `;

        htmlft += `
            <tr>
                <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;">
                    ${d.symbol}
                </td>
                <td>${d.price?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.targetHighPrice?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.targetLowPrice?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.targetMeanPrice?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.recommendationKey || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.customPriceAlertConfidence || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.fiftyTwoWeekRange || '-'}</td>
            </tr>
        `;
    });

    return {htmlt, htmlhl, htmlfr, htmlfg, htmlft}
    
}


function technical_table1(data){
    let html = ""
    data.forEach(d => {
        let sig30 = d.signal_30m || "";
        let sig60 = d.signal_60m || "";
        let sig1d = d.signal_1d || "";

        let color30 =
            sig30.includes("Strong Buy") ? "darkgreen" :
            sig30.includes("Strong Sell") ? "darkred" :
            sig30.includes("Buy") ? "lightgreen" :
            sig30.includes("Sell") ? "red" : "black";
        let color60 =
            sig60.includes("Strong Buy") ? "darkgreen" :
            sig60.includes("Strong Sell") ? "darkred" :
            sig60.includes("Buy") ? "green" :
            sig60.includes("Sell") ? "red" : "black";
        let color1d =
            sig1d.includes("Strong Buy") ? "darkgreen" :
            sig1d.includes("Strong Sell") ? "darkred" :
            sig1d.includes("Buy") ? "green" :
            sig1d.includes("Sell") ? "red" : "black";

        html += `
            <tr>
                <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;
                background:${
                    d.is_fut
                    ? '#fff3cd'
                    : ''
                };

                font-weight:${
                    d.is_fut
                    ? 'bold'
                    : 'normal'
                };">
                    ${d.symbol}
                </td>
                
                <td>${d.industry || '-'}</td>
                <td class="ltp-cell"
                    data-symbol="${d.symbol}"
                    data-ltp="${d.ltp || 0}"
                    style="font-weight:bold;">
                    ${d.ltp?.toFixed(2) || '-'}
                </td>
                <td style="${getReturnColor(d.ret1)}">${d.ret1?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret5)}">${d.ret5?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret15)}">${d.ret15?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret30)}">${d.ret30?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret90)}">${d.ret90?.toFixed(2) || '-'}</td>
                <td style="color:${color30}; font-weight:bold;">${sig30 || '-'}</td>
                <td style="color:${color60}; font-weight:bold;">${sig60 || '-'}</td>
                <td style="color:${color1d}; font-weight:bold;">${sig1d || '-'}</td>
                <td style="${getRSIColor(d.rsi_30m)}">${d.rsi_30m || '-'}</td>
                <td style="${getRSIColor(d.rsi_60m)}">${d.rsi_60m || '-'}</td>
                <td style="${getRSIColor(d.rsi_1d)}">${d.rsi_1d || '-'}</td>
                <td>${d.price_tenkan_30m || '-'}</td>
                <td>${d.price_tenkan_60m || '-'}</td>
                <td>${d.price_tenkan_1d || '-'}</td>
                <td>${d.tenkan_kijun_30m || '-'}</td>
                <td>${d.tenkan_kijun_60m || '-'}</td>
                <td>${d.tenkan_kijun_1d || '-'}</td>
            </tr>
        `;
    });
    return html
}
function highlow_table(data){
    let html = ""
    data.forEach(d => {
        html +=`
            <tr>
                <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;
                background:${
                    d.is_fut
                    ? '#fff3cd'
                    : ''
                };

                font-weight:${
                    d.is_fut
                    ? 'bold'
                    : 'normal'
                };">
                    ${d.symbol}
                </td>
                <td class="ltp-cell"
                    data-symbol="${d.symbol}"
                    data-ltp="${d.ltp || 0}"
                    style="font-weight:bold;">
                    ${d.ltp?.toFixed(2) || '-'}
                </td>
                <td style="${getReturnColor(d.ret1)}">${d.ret1?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retdayHigh)}">${d.retdayHigh?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retdayLow)}">${d.retdayLow?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retweekHigh)}">${d.retweekHigh?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retweekLow)}">${d.retweekLow?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retmonthHigh)}">${d.retmonthHigh?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retmonthLow)}">${d.retmonthLow?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retyearHigh)}">${d.retyearHigh?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.retyearLow)}">${d.retyearLow?.toFixed(2) || '-'}</td>
            </tr>
        `;
    });
    return html
}
function fundamental_table(data){
    let html = ""
    data.forEach(d => {
        
        html += `
            <tr>
                <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;">
                    ${d.symbol}
                </td>
                <td style="${getFundamentalColor(d.ret1)}">${d.beta?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.trailingPE?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.forwardPE?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.trailingEPS?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.forwardEPS?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.epsCurrentYear?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.epsForward?.toFixed(2) || '-'}</td>
                
                <td style="${getReturnColor(d.ret1)}">${d.quickRatio?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.currentRatio?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.debtToEquity?.toFixed(2) || '-'}</td>
            </tr>
        `;
    });
    // html += `
    //     <tr>
    //         <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
            
    //         <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;">
    //             ${d.symbol}
    //         </td>
    //         <td style="${getReturnColor(d.ret1)}">${d.composite_score?.toFixed(2) || '-'}</td>
    //         <td>${d.composite_label || '-'}</td>
    //         <td style="${getReturnColor(d.ret1)}">${d.valuation_score?.toFixed(2) || '-'}</td>
    //         <td>${d.valuation_label || '-'}</td>
    //         <td style="${getReturnColor(d.ret1)}">${d.growth_score?.toFixed(2) || '-'}</td>
    //         <td>${d.growth_label || '-'}</td>
    //         <td style="${getReturnColor(d.ret1)}">${d.profitability_score?.toFixed(2) || '-'}</td>
    //         <td>${d.profitability_label || '-'}</td>
    //         <td style="${getReturnColor(d.ret1)}">${d.risk_score?.toFixed(2) || '-'}</td>
    //         <td>${d.risk_label || '-'}</td>
    //         <td style="${getReturnColor(d.ret1)}">${d.sentiment_score?.toFixed(2) || '-'}</td>
    //         <td>${d.sentiment_label || '-'}</td>
    //     </tr>
    // `;
    return html
}
function fundamentalgrowth_table(data){
    let html = ""
    data.forEach(d => {
        
        html += `
            <tr>
                <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;">
                    ${d.symbol}
                </td>
                <td style="${getFundamentalColor(d.ret1)}">${d.earningsQuarterlyGrowth?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.earningsGrowth?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.revenueGrowth?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.revenuePerShare?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.totalCashPerShare?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.profitMargins?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.grossMargins?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.ebitdaMargins?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.enterpriseToRevenue?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.enterpriseToEbitda?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.priceToBook?.toFixed(2) || '-'}</td>
            </tr>
        `;
    });
    return html
}
function priceTarget_table(data){
    let html = ""
    data.forEach(d => {
        
        html += `
            <tr>
                <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;">
                    ${d.symbol}
                </td>
                <td>${d.price?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.targetHighPrice?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.targetLowPrice?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.targetMeanPrice?.toFixed(2) || '-'}</td>
                <td style="${getFundamentalColor(d.ret1)}">${d.recommendationKey || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.customPriceAlertConfidence || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.fiftyTwoWeekRange || '-'}</td>
            </tr>
        `;
    });
    return html
}


function popup_table(data){
    let html = ""
    data.forEach(d => {
        let sig30 = d.signal_30m || "";
        let sig60 = d.signal_60m || "";
        let sig1d = d.signal_1d || "";

        let color30 =
            sig30.includes("Strong Buy") ? "darkgreen" :
            sig30.includes("Strong Sell") ? "darkred" :
            sig30.includes("Buy") ? "lightgreen" :
            sig30.includes("Sell") ? "red" : "black";
        let color60 =
            sig60.includes("Strong Buy") ? "darkgreen" :
            sig60.includes("Strong Sell") ? "darkred" :
            sig60.includes("Buy") ? "green" :
            sig60.includes("Sell") ? "red" : "black";
        let color1d =
            sig1d.includes("Strong Buy") ? "darkgreen" :
            sig1d.includes("Strong Sell") ? "darkred" :
            sig1d.includes("Buy") ? "green" :
            sig1d.includes("Sell") ? "red" : "black";

        html += `
            <tr>
                <td><input type="checkbox" class="modalStockCheckbox" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;">
                    ${d.symbol}
                </td>
                
                <td>${d.price?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.ret1?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret5)}">${d.ret5?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret15)}">${d.ret15?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret30)}">${d.ret30?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret90)}">${d.ret90?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.rs5)}">${d.rs5?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.rs15)}">${d.rs15?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.rs30)}">${d.rs30?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.rs90)}">${d.rs90?.toFixed(2) || '-'}</td>
                <td style="color:${color30}; font-weight:bold;">${sig30 || '-'}</td>
                <td style="color:${color60}; font-weight:bold;">${sig60 || '-'}</td>
                <td style="color:${color1d}; font-weight:bold;">${sig1d || '-'}</td>
            </tr>
        `;
    });
    return html
}
function popup_stock_table(data){
    let html = ""
    data.forEach(d => {
        let sig1d = d.signal_1d || "";


        let color1d =
            sig1d.includes("Strong Buy") ? "darkgreen" :
            sig1d.includes("Strong Sell") ? "darkred" :
            sig1d.includes("Buy") ? "green" :
            sig1d.includes("Sell") ? "red" : "black";

        html += `
            <tr>
                <td><input type="checkbox" class="modalStockCheckbox" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;">
                    ${d.symbol}
                </td>
                
                <td>${d.price?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.ret1?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret5)}">${d.ret5?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret15)}">${d.ret15?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret30)}">${d.ret30?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.ret90)}">${d.ret90?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.rs5)}">${d.rs5?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.rs15)}">${d.rs15?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.rs30)}">${d.rs30?.toFixed(2) || '-'}</td>
                <td style="${getReturnColor(d.rs90)}">${d.rs90?.toFixed(2) || '-'}</td>
            </tr>
        `;
    });
    return html
}