
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


function fundamental_table(data){
    let html = ""
    data.forEach(d => {

        html += `
            <tr>
                <td><input type="checkbox" class="rowCheck" value="${d.symbol}"></td>
                
                <td onclick="openChart('${d.symbol}')" style="cursor:pointer; color:#3498db;">
                    ${d.symbol}
                </td>
                <td style="${getReturnColor(d.ret1)}">${d.composite_score?.toFixed(2) || '-'}</td>
                <td>${d.composite_label || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.valuation_score?.toFixed(2) || '-'}</td>
                <td>${d.valuation_label || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.growth_score?.toFixed(2) || '-'}</td>
                <td>${d.growth_label || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.profitability_score?.toFixed(2) || '-'}</td>
                <td>${d.profitability_label || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.risk_score?.toFixed(2) || '-'}</td>
                <td>${d.risk_label || '-'}</td>
                <td style="${getReturnColor(d.ret1)}">${d.sentiment_score?.toFixed(2) || '-'}</td>
                <td>${d.sentiment_label || '-'}</td>
            </tr>
        `;
    });
    return html
}