
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