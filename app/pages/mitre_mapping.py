"""
MITRE ATT&CK mapping page for the Network Anomaly Detection Platform.
Displays mapping between detected anomalies and MITRE ATT&CK techniques.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from core.mitre_mapper import MitreMapper
from core.auto_analysis import auto_analysis_service
from core.notification_service import notification_service

def show_mitre_mapping():
    """Display the MITRE ATT&CK mapping page."""
    
    st.header("🛡️ MITRE ATT&CK Mapping")
    st.markdown("**Automatic mapping of detected anomalies to MITRE ATT&CK techniques and tactics.**")
    
    # Check if anomalies are detected
    if 'anomalies' not in st.session_state or st.session_state.anomalies is None:
        st.info("No anomalies detected. Please run anomaly detection first.")
        return
    
    # Get anomalies
    anomalies = st.session_state.anomalies
    
    # Additional check for empty anomalies
    if isinstance(anomalies, pd.DataFrame) and anomalies.empty:
        st.info("No anomalies detected. Please run anomaly detection first.")
        return
    elif not isinstance(anomalies, pd.DataFrame):
        st.error("Invalid anomalies data. Please re-run anomaly detection.")
        return
    
    # Initialize MITRE mapper
    mitre_mapper = MitreMapper()
    
    # Check if automatic mapping was already done
    auto_mapped = st.session_state.get('mitre_auto_mapped', False)
    existing_mappings = st.session_state.get('mitre_mappings')
    
    if auto_mapped and existing_mappings:
        st.success("🤖 **Automatic MITRE mapping completed!** Results are displayed below.")
        st.info("💡 **Tip:** This mapping was generated automatically during anomaly detection. You can re-run with different settings if needed.")
    else:
        st.info("⚠️ **No automatic mapping found.** Please run the mapping analysis below.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Technique Mapping", "📊 Tactics Overview", "⚙️ Custom Rules"])
    
    # Technique Mapping tab
    with tab1:
        st.subheader("🎯 Map Anomalies to ATT&CK Techniques")
        
        # Display existing mapping results if available
        if existing_mappings:
            st.markdown("### 📋 Current Mapping Results")
            
            # Count techniques and tactics
            technique_counts = mitre_mapper.get_technique_counts(existing_mappings)
            tactic_counts = mitre_mapper.get_tactic_counts(existing_mappings)
            
            # Show summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔍 Mapped Anomalies", len(existing_mappings))
            with col2:
                st.metric("🎯 Unique Techniques", len(technique_counts))
            with col3:
                st.metric("📊 Unique Tactics", len(tactic_counts))
            with col4:
                total_mappings = sum(len(mappings) for mappings in existing_mappings.values())
                st.metric("🔗 Total Mappings", total_mappings)
            
            # Show technique distribution
            if technique_counts:
                st.markdown("#### 🎯 Technique Distribution")
                
                technique_df = pd.DataFrame({
                    'Technique': list(technique_counts.keys()),
                    'Count': list(technique_counts.values())
                })
                
                # Create bar chart
                fig = px.bar(
                    technique_df.head(10), 
                    x='Count', 
                    y='Technique',
                    orientation='h',
                    title="Top 10 MITRE ATT&CK Techniques",
                    color='Count',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Re-mapping section
        with st.expander("🔄 **Re-run Mapping with Different Settings**", expanded=not bool(existing_mappings)):
            st.markdown("Adjust settings and re-run the MITRE mapping analysis.")
            
            # Confidence threshold
            confidence = st.slider(
                "Confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Minimum confidence level for technique mappings"
            )
        
        # Map anomalies button
        if st.button("Map Anomalies to Techniques"):
            with st.spinner("Mapping anomalies to MITRE ATT&CK techniques..."):
                try:
                    # Show debug info
                    st.write("**Debug Info:**")
                    st.write(f"- Number of anomalies: {len(anomalies)}")
                    st.write(f"- Anomaly columns: {list(anomalies.columns)}")
                    st.write(f"- Sample anomaly data: {anomalies.iloc[0].to_dict() if len(anomalies) > 0 and not anomalies.empty else 'No anomalies'}")
                    
                    # Map anomalies to techniques
                    mapping_results = mitre_mapper.map_anomalies(anomalies, confidence)
                    
                    # Store results in session state
                    st.session_state.mitre_mappings = mapping_results
                    
                    # Show results
                    if mapping_results:
                        st.success(f"Mapped {len(mapping_results)} anomalies to MITRE ATT&CK techniques.")
                        st.write(f"**Mapping Results Summary:**")
                        for idx, mappings in mapping_results.items():
                            st.write(f"- Anomaly {idx}: {len(mappings)} techniques mapped")
                    else:
                        st.warning("No mappings found with the current confidence threshold.")
                        st.write("**Possible reasons:**")
                        st.write("- Confidence threshold too high")
                        st.write("- No matching protocols, ports, or patterns found")
                        st.write("- Anomaly data format not recognized")
                        
                        # Show what data we're trying to map
                        if len(anomalies) > 0 and not anomalies.empty:
                            sample = anomalies.iloc[0]
                            st.write("**Sample anomaly for debugging:**")
                            st.json(sample.to_dict())
                        else:
                            st.write("**No anomalies available for mapping.**")
                            
                except Exception as e:
                    st.error(f"Error during mapping: {str(e)}")
                    st.exception(e)
        
        # Display mapping results if available
        if 'mitre_mappings' in st.session_state and st.session_state.mitre_mappings:
            mappings = st.session_state.mitre_mappings
            
            # Count techniques and tactics
            technique_counts = mitre_mapper.get_technique_counts(mappings)
            tactic_counts = mitre_mapper.get_tactic_counts(mappings)
            
            # Show technique counts
            st.subheader("Technique Distribution")
            
            # Create bar chart of technique counts
            technique_df = pd.DataFrame({
                'Technique': list(technique_counts.keys()),
                'Count': list(technique_counts.values())
            }).sort_values('Count', ascending=False)
            
            fig = px.bar(
                technique_df,
                x='Count',
                y='Technique',
                orientation='h',
                title="MITRE ATT&CK Techniques Detected",
                color='Count',
                color_continuous_scale='Reds'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show mapped anomalies
            st.subheader("Mapped Anomalies")
            
            # Create a DataFrame with mapping details
            mapped_data = []
            
            for anomaly_idx, techniques in mappings.items():
                anomaly = anomalies.loc[anomaly_idx]
                
                for technique_mapping in techniques:
                    technique = technique_mapping["technique"]
                    
                    mapped_data.append({
                        'Anomaly Index': anomaly_idx,
                        'Technique ID': technique.get("technique_id"),
                        'Technique Name': technique.get("name"),
                        'Tactic': technique.get("tactic"),
                        'Confidence': technique_mapping["confidence"],
                        'Reason': technique_mapping["reason"],
                        'Anomaly Score': anomaly.get("anomaly_score"),
                        'Source IP': anomaly.get("ip_src") if "ip_src" in anomaly else "",
                        'Destination IP': anomaly.get("ip_dst") if "ip_dst" in anomaly else "",
                        'Protocol': anomaly.get("_ws_col_Protocol") if "_ws_col_Protocol" in anomaly else ""
                    })
            
            if mapped_data:
                mapped_df = pd.DataFrame(mapped_data)
                st.dataframe(mapped_df, use_container_width=True)
                
                # Allow downloading the mapping results
                csv_data = mapped_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Mapping Results",
                    data=csv_data,
                    file_name=f"mitre_mapping_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No mappings to display.")
        
        # Technique details expander
        with st.expander("Technique Details", expanded=False):
            # Get all techniques
            all_techniques = mitre_mapper.get_all_techniques()
            
            # Create a selectbox for techniques
            technique_options = [f"{t['technique_id']}: {t['name']}" for t in all_techniques]
            selected_technique = st.selectbox(
                "Select technique for details",
                options=technique_options
            )
            
            if selected_technique:
                # Extract technique ID
                technique_id = selected_technique.split(":")[0].strip()
                
                # Get technique details
                details = mitre_mapper.get_technique_details(technique_id)
                
                if details:
                    st.markdown(f"## {details['technique_id']}: {details['technique_name']}")
                    st.markdown(f"**Tactic:** {details['tactic']} ({details['tactic_id']})")
                    st.markdown("### Description")
                    st.markdown(details['technique_description'])
                    st.markdown("### Tactic Description")
                    st.markdown(details['tactic_description'])
                else:
                    st.warning("Technique details not found.")
    
    # Tactics Overview tab
    with tab2:
        st.subheader("Tactics Overview")
        
        # Show tactic distribution if mappings available
        if 'mitre_mappings' in st.session_state and st.session_state.mitre_mappings:
            mappings = st.session_state.mitre_mappings
            tactic_counts = mitre_mapper.get_tactic_counts(mappings)
            
            if tactic_counts:
                # Create pie chart of tactic distribution
                tactic_df = pd.DataFrame({
                    'Tactic': list(tactic_counts.keys()),
                    'Count': list(tactic_counts.values())
                })
                
                fig = px.pie(
                    tactic_df,
                    values='Count',
                    names='Tactic',
                    title="MITRE ATT&CK Tactics Distribution",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show tactic details
                st.subheader("Tactics Detected")
                
                # Get all tactics
                all_tactics = mitre_mapper.get_all_tactics()
                
                # Create a table of detected tactics with descriptions
                tactic_details = []
                
                for tactic_name in tactic_counts.keys():
                    for tactic in all_tactics:
                        if tactic['name'] == tactic_name:
                            tactic_details.append({
                                'Tactic ID': tactic['tactic_id'],
                                'Tactic Name': tactic['name'],
                                'Description': tactic['description'],
                                'Count': tactic_counts[tactic_name]
                            })
                            break
                
                if tactic_details:
                    tactic_details_df = pd.DataFrame(tactic_details)
                    st.dataframe(tactic_details_df, use_container_width=True)
            else:
                st.info("No tactics mapped yet.")
        else:
            # Show all available tactics if no mappings
            all_tactics = mitre_mapper.get_all_tactics()
            
            if all_tactics:
                st.subheader("Available Tactics")
                
                tactics_df = pd.DataFrame([{
                    'Tactic ID': t['tactic_id'],
                    'Tactic Name': t['name'],
                    'Description': t['description']
                } for t in all_tactics])
                
                st.dataframe(tactics_df, use_container_width=True)
            else:
                st.warning("No tactic information available.")
    
    # Custom Rules tab
    with tab3:
        st.subheader("Custom Mapping Rules")
        
        # Create a form for adding custom rules
        with st.form("custom_rule_form"):
            # Rule type
            rule_type = st.selectbox(
                "Rule type",
                options=["Protocol", "Port", "Pattern"],
                index=0
            )
            
            # Map rule types to internal keys
            rule_type_map = {
                "Protocol": "protocol_rules",
                "Port": "port_rules",
                "Pattern": "pattern_rules"
            }
            
            # Key input
            key_label = "Protocol" if rule_type == "Protocol" else "Port number" if rule_type == "Port" else "Pattern"
            key_input = st.text_input(f"{key_label}")
            
            # Get all techniques for selection
            all_techniques = mitre_mapper.get_all_techniques()
            technique_options = [f"{t['technique_id']}: {t['name']}" for t in all_techniques]
            
            # Technique selection
            selected_techniques = st.multiselect(
                "Select techniques to map",
                options=technique_options
            )
            
            # Submit button
            submit_rule = st.form_submit_button("Add Custom Rule")
        
        if submit_rule:
            if not key_input:
                st.error(f"Please enter a {key_label.lower()}.")
            elif not selected_techniques:
                st.error("Please select at least one technique.")
            else:
                # Extract technique IDs
                technique_ids = [t.split(":")[0].strip() for t in selected_techniques]
                
                # Convert port to integer if needed
                if rule_type == "Port":
                    try:
                        key_input = int(key_input)
                    except ValueError:
                        st.error("Port must be a number.")
                        key_input = None
                
                if key_input is not None:
                    # Add custom rule
                    success = mitre_mapper.add_custom_rule(
                        rule_type_map[rule_type],
                        key_input,
                        technique_ids
                    )
                    
                    if success:
                        st.success(f"Added custom rule: {rule_type} '{key_input}' mapped to {len(technique_ids)} techniques.")
                        
                        # Save custom mappings
                        mitre_mapper.save_custom_mappings()
                    else:
                        st.error("Failed to add custom rule.")
        
        # Show current custom rules
        st.subheader("Current Custom Rules")
        
        # Protocol rules
        st.markdown("#### Protocol Rules")
        protocol_rules = mitre_mapper.mapping_rules.get("protocol_rules", {})
        if protocol_rules:
            protocol_df = pd.DataFrame([
                {"Protocol": proto, "Technique IDs": ", ".join(ids)}
                for proto, ids in protocol_rules.items()
            ])
            st.dataframe(protocol_df, use_container_width=True)
        else:
            st.info("No custom protocol rules defined.")
        
        # Port rules
        st.markdown("#### Port Rules")
        port_rules = mitre_mapper.mapping_rules.get("port_rules", {})
        if port_rules:
            port_df = pd.DataFrame([
                {"Port": port, "Technique IDs": ", ".join(ids)}
                for port, ids in port_rules.items()
            ])
            st.dataframe(port_df, use_container_width=True)
        else:
            st.info("No custom port rules defined.")
        
        # Pattern rules
        st.markdown("#### Pattern Rules")
        pattern_rules = mitre_mapper.mapping_rules.get("pattern_rules", {})
        if pattern_rules:
            pattern_df = pd.DataFrame([
                {"Pattern": pattern, "Technique IDs": ", ".join(ids)}
                for pattern, ids in pattern_rules.items()
            ])
            st.dataframe(pattern_df, use_container_width=True)
        else:
            st.info("No custom pattern rules defined.")
        
        # Button to reset rules
        if st.button("Reset to Default Rules"):
            # Reinitialize mapping rules
            mitre_mapper.mapping_rules = mitre_mapper._initialize_mapping_rules()
            mitre_mapper.save_custom_mappings()
            st.success("Reset to default mapping rules.")
            st.rerun()